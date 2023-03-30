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
        token = self.kowalski.instances["default"]["token"]
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

        # the response should be like:
        # {'default': {'status': 'success', 'message': 'Successfully executed query', 'data': {'ZTF_alerts': {'ZTF17aaaaaas': [{'objectId': 'ZTF17aaaaaas'}]}}}}

        assert "default" in response
        data = response["default"].get("data")
        assert catalog in data
        assert obj_id in data[catalog]
        assert len(data[catalog][obj_id]) > 0

    def test_query_cone_search_multiple_catalogs(self):
        catalog_1 = "ZTF_alerts"
        catalog_2 = "PGIR_alerts"
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
                    catalog_1: {"filter": {}, "projection": {"_id": 0, "objectId": 1}},
                    catalog_2: {"filter": {}, "projection": {"_id": 0, "objectId": 1}},
                },
            },
            "kwargs": {"filter_first": False},
        }

        response = self.kowalski.query(query=query)

        # the response should be like:
        # {'default': {'status': 'success', 'message': 'Successfully executed query', 'data': {'ZTF_alerts': {'ZTF17aaaaaas': [{'objectId': 'ZTF17aaaaaas'}]},
        # 'PGIR_alerts': {'ZTF17aaaaaas': []}}}}

        assert "default" in response
        data = response["default"].get("data")
        assert catalog_1 in data
        assert catalog_2 in data
        assert obj_id in data[catalog_1]
        assert obj_id in data[catalog_2]
        assert len(data[catalog_1][obj_id]) > 0
        assert len(data[catalog_2][obj_id]) == 0

    def test_query_cone_search_multiple_catalogs_multiple_instances(self):
        token = self.kowalski.instances["default"]["token"]

        k = Kowalski(token=token, protocol="http", host="localhost", port=4000)

        cfg = {
            "token": token,
            "protocol": "http",  # here we change the protocol
            "host": "127.0.0.1",  # this is the trick we use to test the add() method running only one kowalski instance
            "port": 4000,
        }
        k.add(name="test", cfg=cfg)

        catalog_1 = "ZTF_alerts"
        catalog_2 = "PGIR_alerts"
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
                    catalog_1: {"filter": {}, "projection": {"_id": 0, "objectId": 1}},
                    catalog_2: {"filter": {}, "projection": {"_id": 0, "objectId": 1}},
                },
            },
            "kwargs": {"filter_first": False},
        }

        response = k.query(query=query)

        # the response should be like:
        # {'default': {'status': 'success', 'message': 'Successfully executed query', 'data': {'ZTF_alerts': {'ZTF17aaaaaas': [{'objectId': 'ZTF17aaaaaas'}]},
        # 'PGIR_alerts': {'ZTF17aaaaaas': []}}}}

        assert "default" in response
        data = response["default"].get("data")
        assert catalog_1 in data
        assert catalog_2 in data
        assert obj_id in data[catalog_1]
        assert obj_id in data[catalog_2]
        assert len(data[catalog_1][obj_id]) > 0
        assert len(data[catalog_2][obj_id]) == 0

        k.instances["default"][
            "catalogs"
        ] = (
            []
        )  # we make sure the first instance (default) does not have the catalog we are looking for,
        # this is to test the fallback to the second instance

        response = k.query(query=query)

        # the response should be like:
        # {'test': {'status': 'success', 'message': 'Successfully executed query', 'data': {'ZTF_alerts': {'ZTF17aaaaaas': [{'objectId': 'ZTF17aaaaaas'}]},
        # 'PGIR_alerts': {'ZTF17aaaaaas': []}}}}

        assert "test" in response
        data = response["test"].get("data")
        assert catalog_1 in data
        assert catalog_2 in data
        assert obj_id in data[catalog_1]
        assert obj_id in data[catalog_2]
        assert len(data[catalog_1][obj_id]) > 0
        assert len(data[catalog_2][obj_id]) == 0

        assert "default" not in response

        # now, we test what happens if one instance has the first catalog, but not the second one
        # and the other instance has the second catalog, but not the first one
        k.instances["default"]["catalogs"] = [catalog_1]
        k.instances["test"]["catalogs"] = [catalog_2]

        response = k.query(query=query)

        # the response should be like:
        # {'test': {...}, 'default': {...}}
        # 'PGIR_alerts': {'ZTF17aaaaaas': []}}}}

        assert "default" in response
        data = response["default"].get("data")
        assert catalog_1 in data
        assert obj_id in data[catalog_1]
        assert len(data[catalog_1][obj_id]) > 0

        assert catalog_2 not in data

        assert "test" in response
        data = response["test"].get("data")
        assert catalog_2 in data
        assert obj_id in data[catalog_2]
        assert len(data[catalog_2][obj_id]) == 0

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
        assert "default" in response
        data = response["default"].get("data")
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
        max_n_threads = 8
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

        candidates_in_skymap_per_instance = self.kowalski.query_skymap(
            path,
            cumprob,
            jd_start,
            jd_end,
            catalogs,
            program_ids,
            filter_kwargs,
            projection_kwargs,
            max_n_threads=max_n_threads,
        )

        assert len(candidates_in_skymap_per_instance.keys()) > 0
        assert (
            self.kowalski.instances.keys() == candidates_in_skymap_per_instance.keys()
        )
        for catalog in catalogs:
            assert catalog in candidates_in_skymap_per_instance["default"].keys()
            assert len(candidates_in_skymap_per_instance["default"][catalog]) > 0
            assert all(
                [
                    "isdiffpos" in candidate["candidate"].keys()
                    for candidate in candidates_in_skymap_per_instance["default"][
                        catalog
                    ]
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
            "host": "127.0.0.1",  # this is the trick we use to test the add() method running only one kowalski instance
            "port": 4000,
        }
        k.add(name="test", cfg=cfg)

    def test_get_catalogs(self):
        catalogs = self.kowalski.get_catalogs()
        assert len(catalogs) > 0
        assert "ZTF_alerts" in catalogs

    def test_get_catalogs_when_multiple_instances(self):
        token = self.kowalski.instances["default"]["token"]

        k = Kowalski(token=token, protocol="http", host="localhost", port=4000)
        cfg = {
            "token": token,
            "protocol": "http",  # here we change the protocol
            "host": "127.0.0.1",  # this is the trick we use to test the add() method running only one kowalski instance
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

    def test_query_when_multiple_instances_with_name(self):
        token = self.kowalski.instances["default"]["token"]

        k = Kowalski(token=token, protocol="http", host="localhost", port=4000)
        cfg = {
            "token": token,
            "protocol": "http",  # here we change the protocol
            "host": "127.0.0.1",  # this is the trick we use to test the add() method running only one kowalski instance
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
        assert "test" in response
        data = response["test"].get("data")
        assert len(data) > 0
        assert data[0]["objectId"] == obj_id

    def test_query_when_multiple_instances_no_name(self):
        token = self.kowalski.instances["default"]["token"]

        k = Kowalski(token=token, protocol="http", host="localhost", port=4000)

        cfg = {
            "token": token,
            "protocol": "http",  # here we change the protocol
            "host": "127.0.0.1",  # this is the trick we use to test the add() method running only one kowalski instance
            "port": 4000,
        }
        k.add(name="test", cfg=cfg)
        k.instances["default"][
            "catalogs"
        ] = (
            []
        )  # we make sure the first instance (default) does not have the catalog we are looking for,
        # this is to test the fallback to the second instance

        catalog = "ZTF_alerts"

        query = {
            "query_type": "find",
            "query": {
                "catalog": catalog,
                "filter": {},
                "projection": {"_id": 0, "objectId": 1},
            },
            "kwargs": {
                "max_time_ms": 10000,
            },
        }

        q = k.query(query)
        assert "test" in q
        assert "data" in q["test"]
        assert len(q["test"]["data"]) > 0
        assert (
            len(list(q["test"]["data"][0].keys())) == 1
        )  # we verify that the projection worked
        assert "objectId" in q["test"]["data"][0].keys()
