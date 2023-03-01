""" PENQUINS - Processing ENormous Queries of kowalski Users INStantaneously """

__all__ = ["Kowalski", "__version__"]


import os
from pathlib import Path
import secrets
import string
import traceback
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from netrc import netrc
from typing import List, Mapping, Optional, Sequence, Union

import astropy.units as u
import astropy_healpix as ah
import healpy as hp
import requests
from astropy.io import fits
from astropy.time import Time
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


def get_cones(path, cumprob):
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
    """Class to communicate with a Kowalski instance"""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        protocol: str = "https",
        host: str = "kowalski.caltech.edu",
        port: int = 443,
        verbose: bool = False,
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

        if (username is None) and (password is None) and (token is None):
            raise ValueError(
                "Missing credentials: provide either username and password or token"
            )

        if (username is not None) and (password is None):
            netrc_auth = netrc().authenticators(host)
            if netrc_auth:
                username, _, password = netrc_auth

        # Status, Kowalski!
        self.v = verbose

        self.protocol = protocol

        self.host = host
        self.port = port

        self.base_url = f"{self.protocol}://{self.host}:{self.port}"

        self.session = requests.Session()

        # Prevent Requests from attempting to do HTTP basic auth using a
        # matching username and password from the user's ~/.netrc file,
        # because kowalski will reject all HTTP basic auth attempts.
        self.session.trust_env = False

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
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        # set up authentication headers
        if token is None:
            self.username = username
            self.password = password
            self.token = self.authenticate()
        else:
            self.token = token

        self.headers = {"Authorization": self.token}

        self.methods = {
            "get": self.session.get,
            "post": self.session.post,
            "put": self.session.put,
            "patch": self.session.patch,
            "delete": self.session.delete,
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        # run shut down procedure
        self.close()
        return False

    def close(self):
        """Shutdown session gracefully
        :return:
        """
        try:
            self.session.close()
            return True
        except Exception as e:
            if self.v:
                print(e)
            return False

    def authenticate(self):
        """Authenticate user, return access token
        :return:
        """

        # post username and password, get access token
        auth = self.session.post(
            f"{self.base_url}/api/auth",
            json={
                "username": self.username,
                "password": self.password,
                "penquins.__version__": __version__,
            },
        )

        if auth.status_code == requests.codes.ok:
            if self.v:
                print(auth.json())

            if "token" not in auth.json():
                print("Authentication failed")
                raise Exception(auth.json().get("message", "Authentication failed"))

            access_token = auth.json().get("token")

            if self.v:
                print("Successfully authenticated")

            return access_token

        raise Exception("Authentication failed")

    def api(self, method: str, endpoint: str, data: Optional[Mapping] = None):
        """Call API endpoint on Kowalski"""
        method = method.lower()
        # allow using both "/<path>/<endpoint>" and "<path>/<endpoint>"
        endpoint = endpoint[1:] if endpoint.startswith("/") else endpoint

        if endpoint is None:
            raise ValueError("Endpoint not specified")
        if method not in ["get", "post", "put", "patch", "delete"]:
            raise ValueError(f"Unsupported method: {method}")

        if method != "get":
            resp = self.methods[method](
                os.path.join(self.base_url, endpoint),
                json=data,
                headers=self.headers,
            )
        else:
            resp = self.methods[method](
                os.path.join(self.base_url, endpoint),
                params=data,
                headers=self.headers,
            )

        return loads(resp.text)

    def batch_query(self, queries: Sequence[Mapping], n_treads: int = 4):
        """Call Kowalski's /api/queries endpoint using multiple processes

        :param queries: sequence of queries
        :param n_treads: number of processes to use
        :return:
        """
        n_treads = min(len(queries), n_treads)

        with ThreadPool(processes=n_treads) as pool:
            if self.v:
                return list(tqdm(pool.imap(self.query, queries), total=len(queries)))
            return list(pool.imap(self.query, queries))

    def query(self, query: Mapping):
        """Call Kowalski's /api/queries endpoint using multiple processes

        :param query: query mapping
        :return:
        """
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

        resp = self.session.post(
            os.path.join(self.base_url, "api/queries"),
            json=_query,
            headers=self.headers,
        )

        return loads(resp.text)

    def ping(self) -> bool:
        """Ping Kowalski

        :return: True if connection is ok, False otherwise
        """
        try:
            resp = self.session.get(
                os.path.join(self.base_url, ""),
                headers=self.headers,
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
        path: Path,
        cumprob: float,
        start_date: str,
        end_date: str,
        min_detections: int,
        drb: float,
        catalogs: List[str],
        program_ids: List[int],
        n_treads=6,
    ) -> List[dict]:
        missing_args = [
            arg
            for arg in [
                path,
                cumprob,
                start_date,
                end_date,
                min_detections,
                drb,
                catalogs,
                program_ids,
            ]
            if arg is None
        ]
        if len(missing_args) > 0:
            raise ValueError(f"Missing arguments: {missing_args}")

        cones = get_cones(path, cumprob)
        jd_start = Time(start_date).jd
        jd_end = Time(end_date).jd

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
                            "filter": {
                                "candidate.jd": {"$gt": jd_start, "$lt": jd_end},
                                "candidate.drb": {"$gt": drb},
                                "candidate.ndethist": {"$gt": min_detections},
                                "candidate.jdstarthist": {
                                    "$gt": jd_start,
                                    "$lt": jd_end,
                                },
                                "candidate.programid":
                                # needs to be in the list of program_ids
                                {
                                    "$in": program_ids
                                },  # 1 = ZTF Public, 2 = ZTF Public+Partnership, 3 = ZTF Public+Partnership+Caltech
                            },
                            "projection": {
                                "_id": 0,
                                "candid": 1,
                                "objectId": 1,
                                "candidate.ra": 1,
                                "candidate.dec": 1,
                                "candidate.jd": 1,
                                "candidate.jdendhist": 1,
                                "candidate.magpsf": 1,
                                "candidate.sigmapsf": 1,
                            },
                        }
                        for catalog in catalogs
                    },
                },
            }

            queries.append(query)

        response = self.batch_query(queries=queries, n_treads=n_treads)
        candidates_per_catalogs = {catalog: [] for catalog in catalogs}

        for r in response:
            assert "data" in r
            data = r.get("data")
            for catalog in catalogs:
                assert catalog in data
                candidates_per_catalogs[catalog].extend(data[catalog]["object"])

        candidates_per_catalogs = {
            catalog: list({c["candid"]: c for c in candidates}.values())
            for catalog, candidates in candidates_per_catalogs.items()
        }

        return candidates_per_catalogs
