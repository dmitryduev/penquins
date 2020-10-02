""" PENQUINS - Processing ENormous Queries of kowalski Users INStantaneously """

__all__ = ['Kowalski']


from copy import deepcopy
from bson.json_util import loads
from multiprocessing.pool import ThreadPool
from netrc import netrc
import os
import requests
from requests.adapters import HTTPAdapter, DEFAULT_POOLBLOCK, DEFAULT_POOLSIZE, DEFAULT_RETRIES
import secrets
import string
import time
import traceback
from tqdm.auto import tqdm
from typing import Union


__version__ = '2.0.0'


Num = Union[int, float]
QueryPart = Union['task', 'result']
Method = Union['get', 'post', 'put', 'patch', 'delete']


class Kowalski(object):
    """
        Class to communicate with a Kowalski instance
    """

    def __init__(self, username=None, password=None, token=None,
                 protocol: str = 'https', host: str = 'kowalski.caltech.edu', port: int = 443,
                 pool_connections=None, pool_maxsize=None, max_retries=None, pool_block=None,
                 verbose: bool = False):
        """
            username, password, token, protocol, host, port:
                Kowalski instance access credentials and address
                If password is omitted, then look up default credentials from
                the ~/.netrc file.
            pool_connections, pool_maxsize, max_retries, pool_block:
                control requests.Session connection pool
            verbose:
                "Status, Kowalski!"
        """

        if (username is None) and (password is None) and (token is None):
            raise ValueError("Missing credentials: provide either username and password or token")

        if (username is not None) and (password is None):
            netrc_auth = netrc().authenticators(host)
            if netrc_auth:
                username, _, password = netrc_auth

        # Status, Kowalski!
        self.v = verbose

        self.protocol = protocol

        self.host = host
        self.port = port

        self.base_url = f'{self.protocol}://{self.host}:{self.port}'

        self.session = requests.Session()

        # Prevent Requests from attempting to do HTTP basic auth using a
        # matching username and password from the user's ~/.netrc file,
        # because kowalski will reject all HTTP basic auth attempts.
        self.session.trust_env = False

        # requests' defaults overridden?
        if (pool_connections is not None) or (pool_maxsize is not None) \
                or (max_retries is not None) or (pool_block is not None):

            pc = pool_connections if pool_connections is not None else DEFAULT_POOLSIZE
            pm = pool_maxsize if pool_maxsize is not None else DEFAULT_POOLSIZE
            mr = max_retries if max_retries is not None else DEFAULT_RETRIES
            pb = pool_block if pool_block is not None else DEFAULT_POOLBLOCK

            self.session.mount('https://', HTTPAdapter(pool_connections=pc, pool_maxsize=pm,
                                                       max_retries=mr, pool_block=pb))
            self.session.mount('http://', HTTPAdapter(pool_connections=pc, pool_maxsize=pm,
                                                      max_retries=mr, pool_block=pb))

        if token is None:
            self.username = username
            self.password = password
            self.token = self.authenticate()
        else:
            self.token = token

        self.headers = {'Authorization': self.token}

        self.methods = {
            'get': self.session.get,
            'post': self.session.post,
            'put': self.session.put,
            'patch': self.session.patch,
            'delete': self.session.delete
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

    def authenticate(self, retries: int = 3):
        """Authenticate user, return access token
        :return:
        """

        for retry in range(retries):
            # post username and password, get access token
            auth = self.session.post(
                f'{self.base_url}/api/auth',
                json={"username": self.username, "password": self.password, "penquins.__version__": __version__}
            )

            if auth.status_code == requests.codes.ok:
                if self.v:
                    print(auth.json())

                if 'token' not in auth.json():
                    print('Authentication failed')
                    raise Exception(auth.json().get('message', 'Authentication failed'))

                access_token = auth.json().get('token')

                if self.v:
                    print('Successfully authenticated')

                return access_token

            else:
                if self.v:
                    print('Authentication failed, retrying...')
                # bad status code? sleep before retrying, maybe no connections available due to high load
                time.sleep(0.5)

        raise Exception('Authentication failed')

    def api(self, data: dict, endpoint: str = None, method: Method = None, timeout: Num = 30, retries: int = 3):

        if endpoint is None:
            raise ValueError('Endpoint not specified')
        if method not in ['get', 'post', 'put', 'patch', 'delete']:
            raise ValueError(f'Unsupported method: {method}')

        for retry in range(retries):
            if method.lower() != 'get':
                resp = self.methods[method.lower()](
                    os.path.join(self.base_url, endpoint),
                    json=data, headers=self.headers, timeout=timeout,
                )
            else:
                resp = self.methods[method.lower()](
                    os.path.join(self.base_url, endpoint),
                    params=data, headers=self.headers, timeout=timeout,
                )

            if resp.status_code == requests.codes.ok:
                return loads(resp.text)
            if self.v:
                print('Server response: error')
            # bad status code? sleep before retrying, maybe no connections available due to high load
            time.sleep(0.5)

        raise Exception('API call failed')

    def batch_query(self, queries, n_treads: int = 4):
        n_treads = min(len(queries), n_treads)

        with ThreadPool(processes=n_treads) as pool:
            if self.v:
                return list(tqdm(pool.imap(self.query, queries), total=len(queries)))
            return list(pool.imap(self.query, queries))

    def query(self, query):

        _query = deepcopy(query)

        # by default, all queries are not registered in the db and the task/results are stored on disk as json files
        # giving a significant execution speed up. this behaviour can be overridden.
        save = _query.get('kwargs', dict()).get('save', False)

        if save:
            if 'kwargs' not in _query:
                _query['kwargs'] = dict()
            if '_id' not in _query['kwargs']:
                # generate a unique hash id and store it in query if saving query in db on Kowalski is requested
                _id = ''.join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(32))

                _query['kwargs']['_id'] = _id

        max_retries = 3
        for retry in range(max_retries):
            resp = self.session.post(
                os.path.join(self.base_url, 'api/queries'),
                json=_query,
                headers=self.headers
            )

            if resp.status_code == requests.codes.ok or retry == max_retries - 1:
                return loads(resp.text)
            else:
                # bad status code? sleep before retrying, maybe no connections available due to high load
                time.sleep(0.5)

        return {"status": "error", "message": "Unknown error."}

    def ping(self, timeout: int = 5) -> bool:
        """
            Ping Kowalski
        :return: True if connection ok, False otherwise
        """
        try:
            resp = self.session.get(
                os.path.join(self.base_url, ''),
                headers=self.headers,
                timeout=timeout
            )

            if resp.status_code == requests.codes.ok:
                return True
            return False

        except Exception as _e:
            _err = traceback.format_exc()
            if self.v:
                print(_err)
            return False
