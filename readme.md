# penquins: a python client for [Kowalski](https://github.com/dmitryduev/kowalski-dev)

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

k = Kowalski(
    username=username,
    password=password,
    protocol=protocol,
    host=host,
    port=port
)
```

It is recommended to authenticate once and then just reuse the generated token:

```python
token = k.token
print(token)

k = Kowalski(
    token=token,
    protocol=protocol,
    host=host,
    port=port
)
```

Check connection:

```python
k.ping()
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
data = response.get("data")
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
data = response.get("data")
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

response = k.query(query=query)
data = response.get("data")
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

r = k.query(query=q)
data = r.get("data")
```

Run a batch of queries in parallel:

```python
qs = [
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

rs = k.batch_query(qs, n_treads=4)
```
