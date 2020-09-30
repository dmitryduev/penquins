# penquins: a python client for [Kowalski](https://github.com/dmitryduev/kowalski-dev)

## Quickstart

Install `penquins` from [PyPI](https://pypi.org/project/penquins/):

```bash
pip install penquins --upgrade
```

Connect to a Kowalski instance:

```python
from penquins import Kowalski

username = '<username>'
password = '<password>'

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

Run a `cone_search` query:

```python
q = {
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

r = k.query(q)
data = r.get('data')
```

Run a `find` query:

```python
q = {
    'query_type': 'find',
    'query': {
        'catalog': 'ZTF_alerts',
        'filter': {
            "objectId": "ZTF20acfkzcg"
        },
        "projection": {
            "_id": 0,
            "candid": 1
        }
    }
}

r = k.query(q)
data = r.get('data')
```

Run a batch of queries in parallel:

```python
qs = [
    {
        'query_type': 'find',
        'query': {
            'catalog': 'ZTF_alerts',
            'filter': {
                "candid": alert['candid']
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
