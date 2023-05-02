""" PENQUINS - Processing ENormous Queries of kowalski Users INStantaneously """

__all__ = ["Kowalski", "__version__"]

import os
import secrets
import string
import traceback
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from netrc import netrc
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union, Tuple

import astropy.units as u
import astropy_healpix as ah
import healpy as hp
import requests
from astropy.io import fits
from bson.json_util import loads
from mocpy import MOC
from requests.adapters import DEFAULT_POOLBLOCK, DEFAULT_POOLSIZE, HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from tqdm.auto import tqdm

__version__ = "2.2.1"

Num = Union[int, float]

DEFAULT_TIMEOUT: int = 5  # seconds
DEFAULT_RETRIES: int = 3
DEFAULT_BACKOFF_FACTOR: int = 1

query_type_one_catalog = [
    "find_one",
    "find",
    "schemas",
    "count_documents",
    "estimated_document_count",
]
query_type_multiple_catalogs = ["cone_search", "near"]


def get_cones(path, cumprob):  # path or file-like object
    max_order = None
    with fits.open(path) as hdul:
        hdul[1].columns
        data = hdul[1].data
        max_order = hdul[1].header["MOCORDER"]

    uniq = data["UNIQ"]
    probdensity = data["PROBDENSITY"]

    level, _ = ah.uniq_to_level_ipix(uniq)
    area = ah.nside_to_pixel_area(ah.level_to_nside(level)).to_value(u.steradian)
    prob = probdensity * area

    moc = MOC.from_valued_healpix_cells(uniq, prob, max_order, cumul_to=cumprob)
    moc_json = moc.serialize(format="json")
    cones = []
    for order, values in moc_json.items():
        for value in values:
            ra, dec = hp.pix2ang(2 ** int(order), int(value), lonlat=True, nest=True)
            r = hp.max_pixrad(2 ** int(order), degrees=True)
            cones.append([ra, dec, r])
    return cones


class TimeoutHTTPAdapter(HTTPAdapter):
    def __init__(self, *args, **kwargs):
        self.timeout = DEFAULT_TIMEOUT
        if "timeout" in kwargs:
            self.timeout = kwargs["timeout"]
            del kwargs["timeout"]
        super().__init__(*args, **kwargs)

    def send(self, request, **kwargs):
        try:
            timeout = kwargs.get("timeout")
            if timeout is None:
                kwargs["timeout"] = self.timeout
            return super().send(request, **kwargs)
        except AttributeError:
            kwargs["timeout"] = DEFAULT_TIMEOUT


class Kowalski:
    """Class to communicate with one or many Kowalski instance"""

    multiple_instances = True
    instances = {}

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        protocol: Optional[str] = "https",
        host: Optional[str] = "kowalski.caltech.edu",
        port: Optional[str] = 443,
        verbose: bool = False,
        instances: dict = None,
        **kwargs,
    ):
        """
        username, password, token, protocol, host, port:
            Kowalski instance access credentials and address
            If password is omitted, then look up default credentials from
            the ~/.netrc file.
        timeout, pool_connections, pool_maxsize, max_retries, backoff_factor, pool_block:
            control requests.Session connection pool settings
        verbose:
            "Status, Kowalski!"
        """
        self.instances = {}
        # Status, Kowalski!
        self.v = verbose

        if instances is None:
            # if there is no instances dict, then create one with the "default" instance
            instances = {
                "default": {
                    "username": username,
                    "password": password,
                    "token": token,
                    "protocol": protocol,
                    "host": host,
                    "port": port,
                }
            }
            self.multiple_instances = False

        # verify that there isnt any duplicate instance names, and display an error saying which ones
        if len(instances) != len(set(instances.keys())):
            raise ValueError(
                f"Duplicate instance names: {', '.join([name for name in instances.keys() if instances.keys().count(name) > 1])}"
            )

        for name, cfg in instances.items():
            self.add(name, cfg, **kwargs)

    def add(self, name, cfg, **kwargs):
        # verify that no instance with the same name already exists
        if name in self.instances:
            raise ValueError(f"Instance {name} already exists")

        # verify that no instance has the same host, port, and username + password or token
        for instance_name, instance in self.instances.items():
            if (
                instance["host"] == cfg.get("host", "kowalski.caltech.edu")
                and instance["port"] == cfg.get("port", 443)
                and (
                    (
                        instance.get("username", None) == cfg.get("username", None)
                        and instance.get("password", None) == cfg.get("password", None)
                    )
                    or (instance["token"] == cfg.get("token", None))
                )
            ):
                raise ValueError(
                    f"Instance {name} seems to be a duplicate of {instance_name}"
                )

        self.instances[name] = {}
        try:
            if (
                (cfg.get("username", None) is None)
                and (cfg.get("password", None) is None)
                and (cfg.get("token", None) is None)
            ):
                raise ValueError(
                    f"Missing credentials for {name}: provide either username and password, or token"
                )

            if (cfg.get("username", None) is not None) and (
                cfg.get("password", None) is None
            ):
                netrc_auth = netrc().authenticators(
                    cfg.get("host", "kowalski.caltech.edu")
                )
                if netrc_auth:
                    cfg["username"], _, cfg["password"] = netrc_auth

            self.instances[name]["protocol"] = cfg.get("protocol", "https")
            self.instances[name]["host"] = cfg.get("host", "kowalski.caltech.edu")
            self.instances[name]["port"] = cfg.get("port", 443)

            self.instances[name][
                "base_url"
            ] = f"{self.instances[name]['protocol']}://{self.instances[name]['host']}:{self.instances[name]['port']}"

            # set up session
            self.instances[name]["session"] = requests.Session()
            # Prevent Requests from attempting to do HTTP basic auth using a
            # matching username and password from the user's ~/.netrc file,
            # because kowalski will reject all HTTP basic auth attempts.
            self.instances[name]["session"].trust_env = False
            self.instances[name]["methods"] = {
                "get": self.instances[name]["session"].get,
                "post": self.instances[name]["session"].post,
                "put": self.instances[name]["session"].put,
                "patch": self.instances[name]["session"].patch,
                "delete": self.instances[name]["session"].delete,
            }
            # mount session adapters
            timeout = kwargs.get("timeout", DEFAULT_TIMEOUT)
            pool_connections = kwargs.get("pool_connections", DEFAULT_POOLSIZE)
            pool_maxsize = kwargs.get("pool_maxsize", DEFAULT_POOLSIZE)
            max_retries = kwargs.get("max_retries", DEFAULT_RETRIES)
            backoff_factor = kwargs.get("backoff_factor", DEFAULT_BACKOFF_FACTOR)
            pool_block = kwargs.get("pool_block", DEFAULT_POOLBLOCK)

            retries = Retry(
                total=max_retries,
                backoff_factor=backoff_factor,
                status_forcelist=[405, 429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "PUT", "POST", "PATCH"],
            )
            adapter = TimeoutHTTPAdapter(
                timeout=timeout,
                max_retries=retries,
                pool_connections=pool_connections,
                pool_maxsize=pool_maxsize,
                pool_block=pool_block,
            )
            self.instances[name]["session"].mount("https://", adapter)
            self.instances[name]["session"].mount("http://", adapter)

            # set up authentication headers
            if cfg.get("token", None) is None:
                self.instances[name]["username"] = cfg.get("username", None)
                self.instances[name]["password"] = cfg.get("password", None)
                self.instances[name]["token"] = self.authenticate()
            else:
                self.instances[name]["token"] = cfg.get("token", None)

            self.instances[name]["headers"] = {
                "Authorization": self.instances[name]["token"]
            }
            # if we now have more than one instance, set multiple_instances to True
            if len(self.instances) > 1:
                self.multiple_instances = True

            catalogs = self.get_catalogs(name)
            self.instances[name]["catalogs"] = catalogs

        except Exception as e:
            del self.instances[name]
            if len(self.instances) == 1:
                self.multiple_instances = False
            raise ValueError(f"Failed to add instance {name}: {e}")

    def remove(self, name):
        if name not in list(self.instances.keys()):
            raise ValueError(f"Instance {name} does not exist")
        del self.instances[name]

    def rename(self, name, old_name=None):
        """Rename an instance"""
        if old_name is None:
            if not self.multiple_instances:
                old_name = list(self.instances.keys())[0]
            else:
                raise ValueError(
                    "Please specify instance name when using multiple instances"
                )
        self.instances[name] = self.instances[old_name]
        del self.instances[old_name]

    def close_all(self):
        """Shutdown session(s) gracefully
        :return:
        """
        instances_closed = {}
        for name, instance in self.instances.items():
            if self.v:
                print(f"Shutting down {name}...")
                try:
                    instance["session"].close()
                    instances_closed[name] = True
                except Exception as e:
                    if self.v:
                        print(e)
                    instances_closed[name] = False
        return instances_closed

    def close(self, name=None):
        """Shutdown session gracefully
        :return:
        """
        if name is None:
            if not self.multiple_instances:
                name = list(self.instances.keys())[0]
            else:
                raise ValueError(
                    "Please specify instance name when using multiple instances"
                )

        if self.v:
            print(f"Shutting down {name}...")
        try:
            self.instances[name]["session"].close()
            return True
        except Exception as e:
            if self.v:
                print(e)
            return False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # run shut down procedure
        self.close_all()
        return False

    def authenticate(self, name=None):
        """Authenticate user, return access token
        :return:
        """
        if name is None:
            if not self.multiple_instances:
                name = list(self.instances.keys())[0]
            else:
                raise ValueError(
                    "Please specify instance name when using multiple instances"
                )

        # post username and password, get access token
        auth = self.instances[name]["session"].post(
            f"{self.instances[name]['base_url']}/api/auth",
            json={
                "username": self.instances[name]["username"],
                "password": self.instances[name]["password"],
                "penquins.__version__": __version__,
            },
        )

        if auth.status_code == requests.codes.ok:
            if self.v:
                print(auth.json())

            if "token" not in auth.json():
                print("Authentication failed")
                raise Exception(
                    auth.json().get("message", f"Authentication failed for {name}")
                )

            access_token = auth.json().get("token")

            if self.v:
                print(f"Successfully authenticated to {name}")

            return access_token

        raise Exception(f"Authentication failed for {name}: {str(auth.json())}")

    def ping(self, name=None) -> bool:
        """Ping Kowalski

        :return: True if connection is ok, False otherwise
        """
        if name is None:
            if not self.multiple_instances:
                name = list(self.instances.keys())[0]
            else:
                raise ValueError(
                    "Please specify instance name when using multiple instances"
                )
        try:
            resp = self.instances[name]["session"].get(
                os.path.join(self.instances[name]["base_url"], ""),
                headers=self.instances[name]["headers"],
            )

            if resp.status_code == requests.codes.ok:
                return True
            return False

        except Exception as _e:
            _err = traceback.format_exc()
            if self.v:
                print(_e)
                print(_err)
            return False

    def api(
        self, method: str, endpoint: str, data: Optional[Mapping] = None, name=None
    ):
        """Call API endpoint on Kowalski"""
        if name is None:
            if not self.multiple_instances:
                name = list(self.instances.keys())[0]
            else:
                raise ValueError(
                    "Please specify instance name when using multiple instances"
                )

        method = method.lower()
        # allow using both "/<path>/<endpoint>" and "<path>/<endpoint>"
        endpoint = endpoint[1:] if endpoint.startswith("/") else endpoint

        if endpoint is None:
            raise ValueError("Endpoint not specified")
        if method not in ["get", "post", "put", "patch", "delete"]:
            raise ValueError(f"Unsupported method: {method}")

        if method != "get":
            resp = self.instances[name]["methods"][method](
                os.path.join(self.instances[name]["base_url"], endpoint),
                json=data,
                headers=self.instances[name]["headers"],
            )
        else:
            resp = self.instances[name]["methods"][method](
                os.path.join(self.instances[name]["base_url"], endpoint),
                params=data,
                headers=self.instances[name]["headers"],
            )

        return loads(resp.text)

    def get_catalogs(self, name=None) -> dict:
        if name is None:
            if not self.multiple_instances:
                name = list(self.instances.keys())[0]
            else:
                raise ValueError(
                    "Please specify instance name when using multiple instances"
                )

        query = {
            "query_type": "info",
            "query": {
                "command": "catalog_names",
            },
        }
        response = self.single_query((query, name))
        return response[name].get("data")

    def get_catalogs_all(self) -> dict:
        catalogs = {}
        for name in self.instances.keys():
            catalogs[name] = self.get_catalogs(name=name)
        return catalogs

    def instance_has_catalog(self, catalog: str, name=None) -> bool:
        if name is None:
            if not self.multiple_instances:
                name = list(self.instances.keys())[0]
            else:
                raise ValueError(
                    "Please specify instance name when using multiple instances"
                )
        return catalog in self.instances[name]["catalogs"]

    def prepare_query(self, query, name=None, instances_load=None) -> dict:
        """Based on the catalogs or catalog specified in the query, split the query into multiple queries for each instance"""

        if "catalogs" not in query["query"] and "catalog" not in query["query"]:
            if name is None and not self.multiple_instances:
                name = list(self.instances.keys())[0]
            if name is None:
                raise ValueError(
                    "For catalog-less queries, please specify instance name if using multiple instances"
                )
            return {name: query}, instances_load

        if (
            query["query_type"] in query_type_one_catalog
            and "catalog" not in query["query"]
        ):
            raise ValueError(
                f"Query type {query['query_type']} requires a single catalog"
            )
        if (
            query["query_type"] in query_type_multiple_catalogs
            and "catalogs" not in query["query"]
        ):
            raise ValueError(
                f"Query type {query['query_type']} requires multiple catalogs key (which can contain just one catalog if you want)"
            )

        if instances_load is None:
            instances_load = {name: 0 for name in self.instances.keys()}

        # now we check if the query is for a single catalog or multiple catalogs
        if "catalogs" in query["query"]:
            catalogs = query["query"]["catalogs"]
        else:
            catalogs = {query["query"]["catalog"]: {}}
            if "projection" in query["query"]:
                catalogs[query["query"]["catalog"]]["projection"] = query["query"][
                    "projection"
                ]
            if "filter" in query["query"]:
                catalogs[query["query"]["catalog"]]["filter"] = query["query"]["filter"]

        queries = {name: None for name in self.instances.keys()}
        if name is None:
            # if no name is specified, but we have only one instance, we use that instance
            if not self.multiple_instances:
                name = list(self.instances.keys())[0]
                if not all(
                    [
                        self.instance_has_catalog(catalog, name=name)
                        for catalog in catalogs.keys()
                    ]
                ):
                    raise ValueError(
                        "One or more catalogs specified in the query are not available in the instance"
                    )
                queries[name] = query
                instances_load[name] += 1
            # if no name is specified and we have multiple instances, we split the query into multiple queries by instance based on the catalogs
            elif len(catalogs) == 1:
                # if we have just one catalog, we can just use the original query but first we check in which instance it is available
                catalog = list(catalogs.keys())[0]
                instances_having_catalog = {
                    instance_name: self.instance_has_catalog(
                        catalog, name=instance_name
                    )
                    for instance_name in self.instances.keys()
                }
                if not any(instances_having_catalog.values()):
                    raise ValueError(
                        f"Catalog {catalog} is not available in any instance"
                    )
                # we pick an instance that has the catalog
                instance_name = min(
                    # keep only the instances that have the catalog (True)
                    {k: v for k, v in instances_having_catalog.items() if v is True},
                    key=lambda k: instances_load[k],
                )
                instances_load[instance_name] += 1
                queries[instance_name] = query
                instances_load[instance_name] += 1
            else:
                for catalog in catalogs:
                    instances_having_catalog = {
                        instance_name: self.instance_has_catalog(
                            catalog, name=instance_name
                        )
                        for instance_name in self.instances.keys()
                    }
                    if not any(instances_having_catalog.values()):
                        raise ValueError(
                            f"Catalog {catalog} is not available in any instance"
                        )

                    # we want to split the load between the instances, so we pick the instance with the least load that has the catalog
                    instance_name = min(
                        # keep only the instances that have the catalog (True)
                        {
                            k: v
                            for k, v in instances_having_catalog.items()
                            if v is True
                        },
                        key=lambda k: instances_load[k],
                    )
                    instances_load[instance_name] += 1

                    if queries[instance_name] is None:
                        # we set it to a copy of the original query, but with only the current catalog
                        queries[instance_name] = deepcopy(query)
                        # if it has a single "catalog" key, we remove it as we will replace it with a "catalogs" key
                        if "catalog" in queries[instance_name]["query"]:
                            del queries[instance_name]["query"]["catalog"]
                            if "projection" in queries[instance_name]["query"]:
                                del queries[instance_name]["query"]["projection"]
                            if "filter" in queries[instance_name]["query"]:
                                del queries[instance_name]["query"]["filter"]
                        queries[instance_name]["query"]["catalogs"] = {
                            catalog: catalogs[catalog]
                        }
                    else:
                        # if the instance already has a query, we add the current catalog to the list of catalogs
                        queries[instance_name]["query"]["catalogs"][catalog] = catalogs[
                            catalog
                        ]
        else:
            # if a name is specified, we check if the instance has all the catalogs
            if not all(
                [self.instance_has_catalog(catalog, name=name) for catalog in catalogs]
            ):
                raise ValueError(
                    "One or more catalogs specified in the query are not available in the specified instance"
                )
            queries[name] = query
            instances_load[name] += 1

        return queries, instances_load

    def prepare_queries(self, queries, name=None) -> dict:
        """Based on the catalogs or catalog specified in the queries, split the queries into multiple queries for each instance"""
        queries_per_instance = {name: [] for name in self.instances.keys()}
        instances_load = {name: 0 for name in self.instances.keys()}
        for query in queries:
            query_per_instance, instances_load = self.prepare_query(
                query, name=name, instances_load=instances_load
            )

            for instance_name, query in query_per_instance.items():
                if query is not None:
                    queries_per_instance[instance_name].append(query)

        return queries_per_instance

    def batch_query(self, queries: Sequence[Mapping], n_threads: int = 4):
        """Call Kowalski's /api/queries endpoint using multiple processes

        :param queries: sequence of queries
        :param n_threads: number of processes to use
        :return:
        """
        queries_name_tpl = []
        for name, queries in queries.items():
            for query in queries:
                queries_name_tpl.append((query, name))

        min_threads = max(1, len(queries_name_tpl))
        n_threads = min(min_threads, n_threads)

        with ThreadPool(processes=n_threads) as pool:
            if self.v:
                return list(
                    tqdm(
                        pool.imap(
                            self.single_query,
                            queries_name_tpl,
                            # chunksize=len(queries_name_tpl),
                        )
                    )
                )
            return list(pool.imap(self.single_query, queries_name_tpl))

    def single_query(self, query_tpl: Tuple[Mapping, str]):
        """Call Kowalski's /api/queries endpoint using multiple processes

        :param query: query mapping
        :return:
        """
        query, name = query_tpl
        _query = deepcopy(query)

        # by default, all queries are not registered in the db and the task/results are stored on disk as json files
        # giving a significant execution speed up. this behaviour can be overridden.
        save = _query.get("kwargs", dict()).get("save", False)

        if save:
            if "kwargs" not in _query:
                _query["kwargs"] = dict()
            if "_id" not in _query["kwargs"]:
                # generate a unique hash id and store it in query if saving query in db on Kowalski is requested
                _id = "".join(
                    secrets.choice(string.ascii_lowercase + string.digits)
                    for _ in range(32)
                )

                _query["kwargs"]["_id"] = _id

        resp = self.instances[name]["session"].post(
            os.path.join(self.instances[name]["base_url"], "api/queries"),
            json=_query,
            headers=self.instances[name]["headers"],
        )
        return {name: loads(resp.text)}

    def query(
        self,
        query=None,
        queries: list = None,
        name=None,
        use_batch_query: bool = False,
        max_n_threads: int = 4,
    ):
        """Call Kowalski's /api/queries endpoint using multiple processes

        :param query: query mapping
        :return:
        """
        if query is None and queries is None:
            raise ValueError("Please specify query or queries")
        if query is not None and queries is not None:
            raise ValueError("Please specify either query or queries, not both")
        if query is not None:
            query_split_in_queries, _ = self.prepare_query(query, name=name)

            if name is not None:
                return self.single_query((query_split_in_queries[name], name))

            if len(query_split_in_queries) == 1:
                return self.single_query(
                    (
                        query_split_in_queries[list(query_split_in_queries.keys())[0]],
                        list(query_split_in_queries.keys())[0],
                    )
                )

            if use_batch_query:
                if len(query_split_in_queries) == 0:
                    raise ValueError(
                        f"No valid query found in {str(query)}\n which yielded {str(query_split_in_queries)}"
                    )
                # this return a dict of instance and one query, but we want a dict of instance and a list of queries for batch_query
                query_split_in_queries = {
                    name: [query] for name, query in query_split_in_queries.items()
                }
                # the n_threads parameter is the number of instances, maxed at max_n_threads
                results = self.batch_query(
                    query_split_in_queries, n_threads=max_n_threads
                )
                # the results are a list of dicts, we want a single dict with the instance name as key and the list of the queries for that instance as value
                final_results = {
                    name: [result[name] for result in results if name in result]
                    for name in query_split_in_queries.keys()
                }
                return final_results

            else:
                results = {}
                for name, query in query_split_in_queries.items():
                    if query is not None:
                        results.update(self.single_query((query, name)))
                return results

        if queries is not None:
            queries_split_in_queries = self.prepare_queries(queries)
            if len(queries_split_in_queries) == 0:
                raise ValueError(
                    f"No valid queries found in {str(queries)}\n which yielded {str(queries_split_in_queries)}"
                )
            if use_batch_query:
                # the n_threads parameter is the number of instances, maxed at max_n_threads
                results = self.batch_query(
                    queries_split_in_queries, n_threads=max_n_threads
                )
                # the results are a list of dicts, we want a single dict with the instance name as key and the list of the queries for that instance as value
                final_results = {
                    name: [result[name] for result in results if name in result]
                    for name in queries_split_in_queries.keys()
                }
                return final_results
            else:
                results = {}
                for name in queries_split_in_queries.keys():
                    for query in queries_split_in_queries[name]:
                        if query is not None:
                            results.update(self.single_query((query, name)))
                return results

    def query_skymap(
            self,
            path: Path,  # path or file-like object
            cumprob: float,
            jd_start: float,
            jd_end: float,
            jdstarthist_start: float,
            jdstarthist_end: float,
            catalogs: List[str],
            program_ids: List[int],
            filter_kwargs: Optional[Mapping] = dict(),
            projection_kwargs: Optional[Mapping] = dict(),
            max_n_threads: int = 4,
    ) -> List[dict]:
        """
        Query Kowalski for objects in a skymap

        :param path: path to skymap file
        :param cumprob: cumulative probability threshold
        :param jd_start: Query candidates detected after this JD
        :param jd_end: Query candidates detected before this JD
        :param jdstarthist_start: Query candidates first detected after this JD
        (i.e. with jdstarthist > jdstarthist_start). This is to ensure sub-threshold
        detections that sometimes show up in jdstarthist are also retrieved.
        :param jdstarthist_end: Query candidates first detected before this JD
        (i.e. with jdstarthist < jdstarthist_end)
        :param catalogs: List of catalogs to query
        :param program_ids: List of program IDs to query
        :param filter_kwargs: Additional filter kwargs
        :param projection_kwargs: Additional projection kwargs
        :param max_n_threads: Maximum number of threads to use
        """
        missing_args = [
            arg
            for arg in [
                path,
                cumprob,
                jd_start,
                jd_end,
                catalogs,
                program_ids,
            ]
            if arg is None
        ]
        if len(missing_args) > 0:
            raise ValueError(f"Missing arguments: {missing_args}")

        cones = get_cones(path, cumprob)

        if isinstance(projection_kwargs, dict):
            if projection_kwargs.get("candid", 1) == 0:
                raise ValueError(
                    "candid cannot be excluded from projection. Do not set it to 0."
                )

        filter = {
            "candidate.jd": {"$gt": jd_start, "$lt": jd_end},
            "candidate.jdstarthist": {
                "$gt": jdstarthist_start,
                "$lt": jdstarthist_end,
            },
            "candidate.programid": {
                "$in": program_ids
            },
            # 1 = ZTF Public, 2 = ZTF Public+Partnership, 3 = ZTF Public+Partnership
            # +Caltech
        }

        for k in filter_kwargs.keys():
            filter[k] = filter_kwargs[k]

        projection = {
            "_id": 0,
            "candid": 1,
            "objectId": 1,
            "candidate.ra": 1,
            "candidate.dec": 1,
            "candidate.jd": 1,
            "candidate.jdendhist": 1,
            "candidate.magpsf": 1,
            "candidate.sigmapsf": 1,
        }

        for k in projection_kwargs.keys():
            projection[k] = projection_kwargs[k]

        queries = []
        for cone in cones:
            query = {
                "query_type": "cone_search",
                "query": {
                    "object_coordinates": {
                        "cone_search_radius": cone[2],
                        "cone_search_unit": "deg",
                        "radec": {"object": [cone[0], cone[1]]},
                    },
                    "catalogs": {
                        catalog: {
                            "filter": filter,
                            "projection": projection,
                        }
                        for catalog in catalogs
                    },
                },
            }

            queries.append(query)

        response = self.query(
            queries=queries, use_batch_query=True, max_n_threads=max_n_threads
        )

        candidates_per_catalogs_per_instance = {
            name: {catalog: [] for catalog in self.instances[name]["catalogs"]}
            for name in list(self.instances.keys())
        }

        # first we have on response per query. Each response contains a dict with one key per instance
        for name in list(response.keys()):
            for response in response[name]:
                data = response.get("data", None)
                if data is None:
                    continue
                for catalog in catalogs:
                    candidates_per_catalogs_per_instance[name][catalog].extend(
                        data[catalog]["object"]
                    )

        for name in candidates_per_catalogs_per_instance.keys():  # remove duplicates
            for catalog in candidates_per_catalogs_per_instance[name].keys():
                candidates_per_catalogs_per_instance[name][catalog] = list(
                    {
                        c["candid"]: c
                        for c in candidates_per_catalogs_per_instance[name][catalog]
                    }.values()
                )

        return candidates_per_catalogs_per_instance
