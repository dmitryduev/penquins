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
from typing import List, Mapping, Optional, Sequence, Union

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

__version__ = "2.2.0"


Num = Union[int, float]


DEFAULT_TIMEOUT: int = 5  # seconds
DEFAULT_RETRIES: int = 3
DEFAULT_BACKOFF_FACTOR: int = 1


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
            instances = {"default": {
                "username": username,
                "password": password,
                "token": token,
                "protocol": protocol,
                "host": host,
                "port": port,
            }}
            self.multiple_instances = False
            
        #verify that there isnt any duplicate instance names, and display an error saying which ones
        if len(instances) != len(set(instances.keys())):
            raise ValueError(f"Duplicate instance names: {', '.join([name for name in instances.keys() if instances.keys().count(name) > 1])}")

        for name, cfg in instances.items():
            self.add(name, cfg, **kwargs)

    # we create the method to add an instance to the class
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
                    (instance.get("username", None) == cfg.get("username", None) and instance.get("password", None) == cfg.get("password", None))
                    or (instance["token"] == cfg.get("token", None))
                )
            ):
                raise ValueError(f"Instance {name} seems to be a duplicate of {instance_name}")
        
        self.instances[name] = {}
        try:
            if (cfg.get("username", None) is None) and (cfg.get("password", None) is None) and (
                    cfg.get("token", None) is None
                ):
                    raise ValueError(
                        f"Missing credentials for {name}: provide either username and password, or token"
                    )
                
            if (cfg.get("username", None) is not None) and (cfg.get("password", None) is None):
                netrc_auth = netrc().authenticators(cfg.get("host", "kowalski.caltech.edu"))
                if netrc_auth:
                    cfg["username"], _, cfg["password"] = netrc_auth

            self.instances[name]["protocol"] = cfg.get("protocol", "https")
            self.instances[name]["host"] = cfg.get("host", "kowalski.caltech.edu")
            self.instances[name]["port"] = cfg.get("port", 443)

            self.instances[name]["base_url"] = f"{self.instances[name]['protocol']}://{self.instances[name]['host']}:{self.instances[name]['port']}"

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
            if (cfg.get("token", None) is None):
                self.instances[name]["username"] = cfg.get("username", None)
                self.instances[name]["password"] = cfg.get("password", None)
                self.instances[name]["token"] = self.authenticate()
            else:
                self.instances[name]["token"] = cfg.get("token", None)

            self.instances[name]["headers"] = {"Authorization": self.instances[name]["token"]}
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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # run shut down procedure
        self.close_all()
        return False

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
                raise ValueError("Please specify instance name when using multiple instances")
        
        if self.v:
            print(f"Shutting down {name}...")
        try:
            self.instances[name]["session"].close()
            return True
        except Exception as e:
            if self.v:
                print(e)
            return False

    def authenticate(self, name=None):
        """Authenticate user, return access token
        :return:
        """
        if name is None:
            if not self.multiple_instances:
                name = list(self.instances.keys())[0]
            else:
                raise ValueError("Please specify instance name when using multiple instances")
        
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
                raise Exception(auth.json().get("message", f"Authentication failed for {name}"))

            access_token = auth.json().get("token")

            if self.v:
                print(f"Successfully authenticated to {name}")

            return access_token

        raise Exception(f"Authentication failed for {name}: {str(auth.json())}")

    def api(self, method: str, endpoint: str, data: Optional[Mapping] = None, name=None):
        """Call API endpoint on Kowalski"""
        if name is None:
            if not self.multiple_instances:
                name = list(self.instances.keys())[0]
            else:
                raise ValueError("Please specify instance name when using multiple instances")
        
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

    def batch_query(self, queries: Sequence[Mapping], n_treads: int = 4, name=None):
        """Call Kowalski's /api/queries endpoint using multiple processes

        :param queries: sequence of queries
        :param n_treads: number of processes to use
        :return:
        """
        if name is None:
            if not self.multiple_instances:
                name = list(self.instances.keys())[0]
            else:
                raise ValueError("Batch query: Please specify instance name when using multiple instances. You can't batch query across instances (yet...)")
            
        n_treads = min(len(queries), n_treads)

        with ThreadPool(processes=n_treads) as pool:
            if self.v:
                return tqdm(pool.starmap(self.query, [(query, name) for query in queries], chunksize=len(queries)))
            return pool.starmap(self.query, [(query, name) for query in queries], chunksize=len(queries))

    def query(self, query: Mapping, name=None):
        """Call Kowalski's /api/queries endpoint using multiple processes

        :param query: query mapping
        :return:
        """
        if name is None:
            if not self.multiple_instances:
                name = list(self.instances.keys())[0]
            else:
                catalog_name = query["query"].get("catalog", None)
                if catalog_name is not None:
                    # find which instance has this catalog
                    for name, instance in self.instances.items():
                        if catalog_name in instance["catalogs"]:
                            print(f"No instance name specified, using: {name} which has catalog: {catalog_name}")
                            break
                    if name is None:
                        raise ValueError(f"No instance has catalog: {catalog_name}")
                else:
                    raise ValueError("Please specify instance name when using multiple instances and no catalog is specified in the query")
        
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
        print(f"Querying {name}...done")
        return loads(resp.text)

    def ping(self, name=None) -> bool:
        """Ping Kowalski

        :return: True if connection is ok, False otherwise
        """
        if name is None:
            if not self.multiple_instances:
                name = list(self.instances.keys())[0]
            else:
                raise ValueError("Please specify instance name when using multiple instances")
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

    def query_skymap(
        self,
        path: Path,  # path or file-like object
        cumprob: float,
        jd_start: float,
        jd_end: float,
        catalogs: List[str],
        program_ids: List[int],
        filter_kwargs: Optional[Mapping] = dict(),
        projection_kwargs: Optional[Mapping] = dict(),
        n_treads: int = 4,
        name=None,
    ) -> List[dict]:
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
        
        if name is None:
            if not self.multiple_instances:
                name = list(self.instances.keys())[0]
            else:
                raise ValueError("Please specify instance name when using multiple instances")

        cones = get_cones(path, cumprob)

        if isinstance(projection_kwargs, dict):
            if projection_kwargs.get("candid", 1) == 0:
                raise ValueError(
                    "candid cannot be excluded from projection. Do not set it to 0."
                )

        filter = {
            "candidate.jd": {"$gt": jd_start, "$lt": jd_end},
            "candidate.jdstarthist": {
                "$gt": jd_start,
                "$lt": jd_end,
            },
            "candidate.jdendhist": {
                "$gt": jd_start,
                "$lt": jd_end,
            },
            "candidate.programid": {
                "$in": program_ids
            },  # 1 = ZTF Public, 2 = ZTF Public+Partnership, 3 = ZTF Public+Partnership+Caltech
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

        response = self.batch_query(queries=queries, n_treads=n_treads, name=name)
        candidates_per_catalogs = {catalog: [] for catalog in catalogs}

        for r in response:
            data = r.get("data", None)
            if data is None:
                continue
            for catalog in catalogs:
                candidates_per_catalogs[catalog].extend(data[catalog]["object"])

        if not all(
            [
                len(candidates_per_catalogs[catalog]) > 0
                for catalog in candidates_per_catalogs.keys()
            ]
        ):
            candidates_per_catalogs = {
                catalog: list({c["candid"]: c for c in candidates}.values())
                for catalog, candidates in candidates_per_catalogs.items()
            }  # remove duplicates

        return candidates_per_catalogs

    def get_catalogs(self, name=None) -> dict:
        if name is None:
            if not self.multiple_instances:
                name = list(self.instances.keys())[0]
            else:
                raise ValueError("Please specify instance name when using multiple instances")
        
        query = {
            "query_type": "info",
            "query": {
                "command": "catalog_names",
            },
        }
        response = self.query(query=query, name=name)
        return response.get("data")
    
    def get_catalogs_all(self) -> dict:
        catalogs = {}
        for name in self.instances.keys():
            catalogs[name] = self.get_catalogs(name=name)
        return catalogs
    
