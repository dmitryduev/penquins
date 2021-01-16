import pytest

from penquins import Kowalski


@pytest.fixture(autouse=True, scope="class")
def kowalski(request):
    # from Kowalski's default config:
    username, password, protocol, host, port = (
        "admin",
        "admin",
        "http",
        "localhost",
        4000,
    )

    request.cls.kowalski = Kowalski(
        username=username, password=password, protocol=protocol, host=host, port=port
    )


class TestPenquins:
    """
    Test penquins against a live Kowalski instance running on localhost using the default config
    Meant to be run on GA CI
    """

    def test_token_authorization(self):
        token = self.kowalski.token

        k = Kowalski(token=token, protocol="http", host="localhost", port=4000)

        assert k.ping()

    def test_query_cone_search(self):
        catalog = "ZTF_alerts"
        obj_id = "ZTF17aaaaaas"

        query = {
            "query_type": "cone_search",
            "query": {
                "object_coordinates": {
                    "cone_search_radius": 2,
                    "cone_search_unit": "arcsec",
                    "radec": {
                        obj_id: [
                            68.578209,
                            49.0871395,
                        ]
                    },
                },
                "catalogs": {
                    catalog: {"filter": {}, "projection": {"_id": 0, "objectId": 1}}
                },
            },
            "kwargs": {"filter_first": False},
        }

        response = self.kowalski.query(query=query)
        assert "data" in response
        data = response.get("data")
        assert catalog in data
        assert obj_id in data[catalog]
        assert len(data[catalog][obj_id]) > 0

    def test_query_find(self):
        catalog = "ZTF_alerts"
        obj_id = "ZTF17aaaaaas"

        query = {
            "query_type": "find",
            "query": {
                "catalog": catalog,
                "filter": {"objectId": obj_id},
                "projection": {"_id": 0, "objectId": 1},
            },
        }

        response = self.kowalski.query(query=query)
        assert "data" in response
        data = response.get("data")
        assert len(data) == 0
        assert data[0]["objectId"] == obj_id
