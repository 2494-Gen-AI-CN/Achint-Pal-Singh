"""API regression tests using unittest and a real local server; no test dependencies."""
import json
import math
import socket
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import uvicorn

import app as api


class RestaurantAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.socket = socket.socket()
        cls.socket.bind(("127.0.0.1", 0))
        cls.base_url = f"http://127.0.0.1:{cls.socket.getsockname()[1]}"
        cls.server = uvicorn.Server(uvicorn.Config(api.app, log_level="error"))
        cls.thread = threading.Thread(target=cls.server.run, kwargs={"sockets": [cls.socket]}, daemon=True)
        cls.thread.start()
        deadline = time.monotonic() + 30
        while not cls.server.started:
            if not cls.thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("Test API failed to start with its checkpoint and dataset")
            time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=10)
        cls.socket.close()

    def request(self, path, payload=None, method=None, headers=None):
        request = Request(self.base_url + path,
                          data=json.dumps(payload).encode() if payload is not None else None,
                          headers={"Content-Type": "application/json", **(headers or {})}, method=method)
        try:
            response = urlopen(request, timeout=10)
        except HTTPError as error:
            response = error
        with response:
            body = response.read()
            data = json.loads(body) if body and "application/json" in response.headers.get("Content-Type", "") else None
            return response.status, data, response.headers

    def test_existing_routes_and_model(self):
        self.assertEqual(self.request("/")[0], 200)
        self.assertTrue(self.request("/health")[1]["model_loaded"])
        self.assertIn("U1014", self.request("/users")[1]["users"])
        self.assertEqual(len(self.request("/restaurants")[1]), 130)
        status, data, _ = self.request("/recommend/U1014?top_k=5")
        self.assertEqual(status, 200)
        self.assertEqual(data["total_recommendations"], 5)
        visited = set(api.ratings.loc[api.ratings.userID == "U1014", "placeID"])
        self.assertFalse(visited.intersection(row["restaurant_id"] for row in data["recommendations"]))
        self.assertTrue(all(math.isfinite(row["predicted_rating"]) for row in data["recommendations"]))
        self.assertEqual(self.request("/recommend/missing")[0], 404)
        self.assertEqual(self.request("/recommend/U1014?top_k=0")[0], 422)

    def test_geography_and_options(self):
        status, data, _ = self.request("/discovery/options")
        self.assertEqual(status, 200)
        self.assertEqual(data["countries"], ["Mexico"])
        self.assertEqual(data["unknown_country_count"], 28)
        self.assertIn("Mexican", data["cuisines"])
        self.assertEqual(data["areas"], ["closed", "open"])

    def test_budget_filters_change_results(self):
        ids = []
        for budget in ("low", "medium", "high"):
            status, data, _ = self.request("/recommend/preferences", {"budget": budget, "top_k": 50})
            self.assertEqual(status, 200)
            self.assertGreater(len(data["recommendations"]), 0)
            self.assertTrue(all(row["price"] == budget for row in data["recommendations"]))
            ids.append({row["restaurant_id"] for row in data["recommendations"]})
        self.assertFalse(ids[0] & ids[1] | ids[1] & ids[2] | ids[0] & ids[2])

    def test_cuisine_and_area_filters(self):
        payload = {"budget": "low", "cuisine": "Mexican", "area": "closed", "top_k": 50}
        status, data, _ = self.request("/recommend/preferences", payload)
        self.assertEqual(status, 200)
        self.assertGreater(len(data["recommendations"]), 0)
        for row in data["recommendations"]:
            self.assertIn("Mexican", row["cuisines"])
            self.assertEqual(row["area"], "closed")
            self.assertEqual(row["price"], "low")

    def test_ranking_matches_historical_evidence(self):
        status, data, _ = self.request("/recommend/preferences", {"top_k": 50})
        self.assertEqual(status, 200)
        scores = [row["score"] for row in data["recommendations"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for row in data["recommendations"]:
            values = api.ratings.loc[api.ratings.placeID == row["restaurant_id"], "rating"] * 2
            expected = (values.sum() + 10 * api.ratings.rating.mean() * 2) / (len(values) + 10)
            self.assertAlmostEqual(row["score"], expected, places=4)
            self.assertEqual(row["rating_count"], len(values))
            self.assertEqual(row["method"], "cold_start_bayesian")
            self.assertTrue(row["explanation"])
        self.assertEqual(data, self.request("/recommend/preferences", {"top_k": 50})[1])

    def test_invalid_inputs(self):
        for payload in ({"budget": "cheap"}, {"budget": None}, {"top_k": 0}, {"top_k": 51},
                        {"top_k": "5"}, {"top_k": True}, {"top_k": 1.5}, {"area": "roof"},
                        {"cuisine": "made_up"}, {"cuisine": ""}, {"unexpected": True}):
            with self.subTest(payload=payload):
                self.assertEqual(self.request("/recommend/preferences", payload)[0], 422)

    def test_empty_results_and_default_limit(self):
        self.assertEqual(self.request("/recommend/preferences", {})[1]["total_recommendations"], 5)
        # Select an actual supported combination with no matching records.
        for cuisine in api.available_cuisines(api.catalog):
            subset = api.catalog.loc[api.catalog.cuisines.apply(lambda values: cuisine in values)]
            if not ((subset.price == "high") & (subset.area == "open")).any():
                status, data, _ = self.request("/recommend/preferences", {
                    "cuisine": cuisine, "budget": "high", "area": "open"})
                self.assertEqual(status, 200)
                self.assertEqual(data["recommendations"], [])
                self.assertEqual(data["total_recommendations"], 0)
                break
        else:
            self.fail("Expected an empty metadata combination in the demo dataset")

    def test_cors_post_preflight(self):
        status, _, headers = self.request("/recommend/preferences", method="OPTIONS", headers={
            "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"})
        self.assertEqual(status, 200)
        self.assertEqual(headers["access-control-allow-origin"], "http://localhost:5173")
        self.assertIn("POST", headers["access-control-allow-methods"])


if __name__ == "__main__":
    unittest.main()
