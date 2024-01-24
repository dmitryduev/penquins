# penquins: a python client for [Kowalski](https://github.com/skyportal/kowalski)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5651471.svg)](https://doi.org/10.5281/zenodo.5651471)

`penquins` is a python client for [Kowalski](https://github.com/skyportal/kowalski), a multi-survey data archive and alert broker for time-domain astronomy.

## Quickstart

Install `penquins` from [PyPI](https://pypi.org/project/penquins/):

```bash
pip install penquins --upgrade
```

Connect to a Kowalski instance:

```python
from penquins import Kowalski

username = "<username>"
password = "<password>"

protocol, host, port = "https", "<host>", 443

kowalski = Kowalski(
    username=username,
    password=password,
    protocol=protocol,
    host=host,
    port=port
)
```
*When connecting to only one instance, it will be labeled as "default". Keep this in mind when retrieving the results of your queries.*

Connect to multiple Kowalski instances:

```python
from penquins import Kowalski

instances = {
    "kowalski": {
        "name": "kowalski",
        "host": "<host>",
        "protocol": "https"
        "port": 443,
        "token": "<token>" # or username and password
    },
    ...
}

kowalski = Kowalski(instances=instances)
```

*When using multiple instances at once, you can specify a single instance to query using its name when calling `query(name=...)`, or no name at all. If no name is provided and the catalog(s) being queried is/are available on multiple instances, penquins will divide the load between instances automagically.*

*When retrieving the results, you'll have to use the instance(s) name instead of "default", or simply iterate over the results by instance and merge the results.*


It is recommended to authenticate once and then just reuse the generated token:

```python
token = kowalski.token
print(token)

kowalski = Kowalski(
    token=token,
    protocol=protocol,
    host=host,
    port=port
)
```

Check connection:

```python
kowalski.ping()
```

### Querying a Kowalski instance

Most users will be interacting with Kowalski using the `Kowalski.query` method.

Retrieve available catalog names:

```python
query = {
    "query_type": "info",
    "query": {
        "command": "catalog_names",
    }
}

response = kowalski.query(query=query)
data = response.get("default").get("data")
```

Query for 7 nearest sources to a sky position, sorted by the spheric distance, with a `near` query:

```python
query = {
    "query_type": "near",
    "query": {
        "max_distance": 2,
        "distance_units": "arcsec",
        "radec": {"query_coords": [281.15902595, -4.4160933]},
        "catalogs": {
            "ZTF_sources_20210401": {
                "filter": {},
                "projection": {"_id": 1},
            }
        },
    },
    "kwargs": {
        "max_time_ms": 10000,
        "limit": 7,
    },
}

response = kowalski.query(query=query)
data = response.get("default").get("data")
```

Retrieve available catalog names:

```python
query = {
    "query_type": "info",
    "query": {
        "command": "catalog_names",
    }
}

response = k.query(query=query)
data = response.get("default").get("data")
```

Query for 7 nearest sources to a sky position, sorted by the spheric distance, with a `near` query:

```python
query = {
    "query_type": "near",
    "query": {
        "max_distance": 2,
        "distance_units": "arcsec",
        "radec": {"query_coords": [281.15902595, -4.4160933]},
        "catalogs": {
            "ZTF_sources_20210401": {
                "filter": {},
                "projection": {"_id": 1},
            }
        },
    },
    "kwargs": {
        "max_time_ms": 10000,
        "limit": 7,
    },
}

response = k.query(query=query)
data = response.get("default").get("data")
```

Run a `cone_search` query:

```python
query = {
    "query_type": "cone_search",
    "query": {
        "object_coordinates": {
            "cone_search_radius": 2,
            "cone_search_unit": "arcsec",
            "radec": {
                "ZTF20acfkzcg": [
                    115.7697847,
                    50.2887778
                ]
            }
        },
        "catalogs": {
            "ZTF_alerts": {
                "filter": {},
                "projection": {
                    "_id": 0,
                    "candid": 1,
                    "objectId": 1
                }
            }
        }
    },
    "kwargs": {
        "filter_first": False
    }
}

response = kowalski.query(query=query)
data = response.get("default").get("data")
```

Run a `find` query:

```python
q = {
    "query_type": "find",
    "query": {
        "catalog": "ZTF_alerts",
        "filter": {
            "objectId": "ZTF20acfkzcg"
        },
        "projection": {
            "_id": 0,
            "candid": 1
        }
    }
}

response = kowalski.query(query=q)
data = response.get("default").get("data")
```

Run a batch of queries in parallel:

```python
queries = [
    {
        "query_type": "find",
        "query": {
            "catalog": "ZTF_alerts",
            "filter": {
                "candid": alert["candid"]
            },
            "projection": {
                "_id": 0,
                "candid": 1
            }
        }
    }
    for alert in data
]

responses = k.query(queries=queries, use_batch_query=True, max_n_threads=4)
```

### Querying multiple instances at once

When using multiple instances at once, you can specify a single instance to query using its name when calling `query(name=...)`, or no name at all. If no name is provided, and the catalog(s) being queried is/are available on multiple instances, penquins will divide the load between instances automagically.

When retrieving the results, you'll have to use the instance(s) name instead of "default", or simply iterate over the results by instance and merge the results.

Any of the queries mentioned for single instance querying also work here.

#### Examples

No instance name specified:

```python
q = {
    "query_type": "find",
    "query": {
        "catalog": "ZTF_alerts",
        "filter": {
            "objectId": "ZTF20acfkzcg"
        },
        "projection": {
            "_id": 0,
            "candid": 1
        }
    }
}
response = kowalski.query(query=q)
data = response.get(<instance_name).get("data") # retrieving data from one instance

# OR

data = [] # or {} depending on the query's expected result, differs by query type
for instance, instance_results in response.items():
    for result in instance_results:
        data.append(result.get('data'))
```

Instance name specified:
```python
q = {
    "query_type": "find",
    "query": {
        "catalog": "ZTF_alerts",
        "filter": {
            "objectId": "ZTF20acfkzcg"
        },
        "projection": {
            "_id": 0,
            "candid": 1
        }
    }
}
response = kowalski.query(query=q, name=<instance_name>)
data = response.get(<instance_name).get("data") # retrieving data from one instance
```

### Interacting with the API

Users can interact with [Kowalski's API](https://kowalski.caltech.edu/docs/api/)
in a more direct way using the `Kowalski.api` method.

Users with admin privileges can add/remove users to/from the system:

```python
username = "noone"
password = "nopas!"
email = "user@caltech.edu"

request = {
  "username": username,
  "password": password,
  "email": email
}

response = kowalski.api(method="post", endpoint="/api/users", data=request)

response = kowalski.api(method="delete", endpoint=f"/api/users/{username}")
```

## Publish new version

Please refer to https://realpython.com/pypi-publish-python-package/
for a detailed guide.

```shell script
pip install bumpversion
export PENQUINS_VERSION=2.3.3

bumpversion --current-version $PENQUINS_VERSION minor setup.py penquins/penquins.py
python setup.py sdist bdist_wheel

twine check dist/*$PENQUINS_VERSION*
twine upload dist/*$PENQUINS_VERSION*

username: __token__
token: <TOKEN>
```
