import os

from astropy.time import Time
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
        username=username,
        password=password,
        protocol=protocol,
        host=host,
        port=port,
        verbose=True,
    )
    print("done with the fixture setup")


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
        print("gonna grab a token from the default instance")
        print(self.kowalski.instances)
        token = self.kowalski.instances["default"]["token"]

        print("now creating a new instance of the class")
        k = Kowalski(token=token, protocol="http", host="localhost", port=4000)
        print("will ping")
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

    def test_query_cone_search_from_skymap(self):
        n_treads = 8
        filename = "localization.fits"
        path = os.path.join(os.path.dirname(__file__), "data", filename)

        cumprob = 0.7
        jd_start = Time("2019-01-01").jd
        jd_end = Time("2020-01-02").jd
        catalogs = ["ZTF_alerts"]
        program_ids = [1]

        filter_kwargs = {
            "candidate.drb": {"$gt": 0.8},
            "candidate.ndethist": {"$gt": 1},
        }

        projection_kwargs = {
            "candidate.isdiffpos": 1,
        }

        candidates_in_skymap = self.kowalski.query_skymap(
            path,
            cumprob,
            jd_start,
            jd_end,
            catalogs,
            program_ids,
            filter_kwargs,
            projection_kwargs,
            n_treads=n_treads,
        )

        assert len(candidates_in_skymap.keys()) > 0
        for catalog in catalogs:
            assert catalog in candidates_in_skymap.keys()
            assert len(candidates_in_skymap[catalog]) > 0
            assert all(
                [
                    "isdiffpos" in candidate["candidate"].keys()
                    for candidate in candidates_in_skymap[catalog]
                ]
            )

    def test_cant_add_duplicate_instance(self):
        token = self.kowalski.instances["default"]["token"]

        k = Kowalski(token=token, protocol="http", host="localhost", port=4000)
        cfg = {
            "token": token,
            "protocol": "https",
            "host": "localhost",
            "port": 4000,
        }
        with pytest.raises(ValueError):
            k.add(name="test", cfg=cfg)

    def test_cant_add_instance_with_same_name(self):
        token = self.kowalski.instances["default"]["token"]

        k = Kowalski(token=token, protocol="http", host="localhost", port=4000)
        cfg = {
            "token": token,
            "protocol": "https",
            "host": "localhost",
            "port": 4000,
        }
        with pytest.raises(ValueError):
            k.add(name="default", cfg=cfg)

    def test_add_instance(self):
        token = self.kowalski.instances["default"]["token"]

        k = Kowalski(token=token, protocol="http", host="localhost", port=4000)
        cfg = {
            "token": token,
            "protocol": "http",
            "host": "127.0.0.1", # this is the trick we use to test the add() method running only one kowalski instance
            "port": 4000,
        }
        k.add(name="test", cfg=cfg)

    def test_query_when_multiple_instances_with_name(self):
        token = self.kowalski.instances["default"]["token"]

        k = Kowalski(token=token, protocol="http", host="localhost", port=4000)
        cfg = {
            "token": token,
            "protocol": "http", # here we change the protocol
            "host": "127.0.0.1", # this is the trick we use to test the add() method running only one kowalski instance
            "port": 4000,
        }
        k.add(name="test", cfg=cfg)

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

        response = k.query(query=query, name="test")
        assert "data" in response
        data = response.get("data")
        assert len(data) > 0
        assert data[0]["objectId"] == obj_id

    def test_get_catalogs(self):
        catalogs = self.kowalski.get_catalogs()
        assert len(catalogs) > 0
        assert "ZTF_alerts" in catalogs

    def test_get_catalogs_when_multiple_instances(self):
        token = self.kowalski.instances["default"]["token"]

        k = Kowalski(token=token, protocol="http", host="localhost", port=4000)
        cfg = {
            "token": token,
            "protocol": "http", # here we change the protocol
            "host": "127.0.0.1", # this is the trick we use to test the add() method running only one kowalski instance
            "port": 4000,
        }
        k.add(name="test", cfg=cfg)

        # get catalogs for one instance
        catalogs = k.get_catalogs(name="test")

        assert len(catalogs) > 0
        assert "ZTF_alerts" in catalogs

        # get catalogs for all instances
        catalogs = k.get_catalogs_all()
        assert set(catalogs.keys()) == set(["default", "test"])
        assert all([len(catalogs[name]) > 0 for name in catalogs.keys()])

