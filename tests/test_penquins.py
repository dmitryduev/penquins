import pytest
import random

from penquins import Kowalski


@pytest.fixture(autouse=True, scope="class")
def kowalski_fixture(request):
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


@pytest.fixture(autouse=True, scope="class")
def user_filter_fixture(request):
    filter_id = random.randint(1, 1000)
    group_id = random.randint(1, 1000)
    collection = "ZTF_alerts"
    permissions = [1, 2]
    pipeline = [
        {
            "$match": {
                "candidate.drb": {"$gt": 0.9999},
                "cross_matches.CLU_20190625.0": {"$exists": False},
            }
        },
        {
            "$addFields": {
                "annotations.author": "dd",
                "annotations.mean_rb": {"$avg": "$prv_candidates.rb"},
            }
        },
        {"$project": {"_id": 0, "candid": 1, "objectId": 1, "annotations": 1}},
    ]

    request.cls.user_filter = {
        "group_id": group_id,
        "filter_id": filter_id,
        "catalog": collection,
        "permissions": permissions,
        "pipeline": pipeline,
    }


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
        assert len(data) > 0
        assert data[0]["objectId"] == obj_id

    def test_api_call_filters(self):
        # post new filter:
        response = self.kowalski.api(
            method="post", endpoint="/api/filters", data=self.user_filter
        )
        assert response["status"] == "success"
        assert "data" in response
        assert "fid" in response["data"]
        fid1 = response["data"]["fid"]

        # retrieve
        filter_id = self.user_filter["filter_id"]
        response = self.kowalski.api(method="get", endpoint=f"/api/filters/{filter_id}")
        assert response["status"] == "success"
        assert response["message"] == f"Retrieved filter id {filter_id}"
        assert "data" in response
        assert "active_fid" in response["data"]
        assert response["data"]["active_fid"] == fid1
        assert "autosave" in response["data"]
        assert not response["data"]["autosave"]

        # turn update_annotations on
        response = self.kowalski.api(
            method="patch",
            endpoint="/api/filters",
            data={
                "filter_id": filter_id,
                "update_annotations": True,
            },
        )
        assert response["status"] == "success"
        assert "data" in response
        assert "update_annotations" in response["data"]
        assert response["data"]["update_annotations"]

        # delete filter
        response = self.kowalski.api(
            method="delete", endpoint=f"/api/filters/{filter_id}"
        )
        assert response["status"] == "success"
        assert response["message"] == f"Removed filter id {filter_id}"
