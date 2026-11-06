import json
import math
import unittest
from unittest.mock import patch
from geo_data import ROOT, explicit_count, explicit_measurement, ingest, query


class AllLayerChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Source-data unit tests must not depend on the operator's live database.
        cls.database_env = patch.dict("os.environ", {"DATABASE_URL": ""})
        cls.database_env.start()
        cls.addClassCleanup(cls.database_env.stop)
        cls.catalog = ingest()

    def test_every_layer_matches_original_counts_and_numeric_total(self):
        for layer in self.catalog["layers"]:
            with self.subTest(layer=layer["source"]):
                features = json.loads((ROOT / layer["source"]).read_text(encoding="utf-8-sig"))["features"]
                self.assertEqual(query(self.catalog, layer["source"], "count")["value"], len(features))
                fields = layer["numeric_fields"]
                if fields:
                    field = "Area" if "Area" in fields else fields[0]
                    values = [f.get("properties", {}).get(field) for f in features]
                    numeric = [v for v in values if isinstance(v, (int, float))
                               and not isinstance(v, bool) and math.isfinite(v)]
                    actual = query(self.catalog, layer["source"], "sum", field=field)
                    self.assertEqual(actual["value"], math.fsum(numeric) if numeric else None)

    def test_direct_counts_and_unknown_fields(self):
        for question, expected in [("How many fire stations are cityside?", 1),
                                   ("How many airside fire stations are there?", 4)]:
            plan = explicit_count(question, self.catalog)
            self.assertEqual(query(self.catalog, **plan)["value"], expected)
        for question in ["How many roads?", "How many trees are there?"]:
            self.assertIsNotNone(explicit_count(question, self.catalog))
        self.assertIsNone(explicit_count("How many hospitals?", self.catalog))
        with self.assertRaises(ValueError):
            query(self.catalog, "Fire Station", "max", field="height")

    def test_argmax_and_duplicate_names(self):
        result = query(self.catalog, "Fire Station", "argmax", field="Area")
        self.assertEqual(result["records"][0]["Name"], "Fire Station 1")
        self.assertEqual(result["value"], 3052.92)
        self.assertEqual(query(self.catalog, "Fire Station", "list", name="Fire Station 99")["records"], [])

    def test_explicit_aggregates_do_not_drop_unknown_filters(self):
        plan = explicit_measurement("What is the total recorded area of fire stations?", self.catalog)
        self.assertEqual(plan["operation"], "sum")
        self.assertEqual(query(self.catalog, **plan)["value"], 8015.735)
        filtered = explicit_measurement("Average area of cityside fire stations", self.catalog)
        self.assertEqual(query(self.catalog, **filtered)["value"], 495.614)
        self.assertIsNone(explicit_measurement("Total area of fire stations taller than 20m", self.catalog))

    def test_name_and_area_projection(self):
        result = query(self.catalog, "Fire Station", "list", fields=["Name", "Area"])
        self.assertEqual(len(result["records"]), 5)
        self.assertEqual(set(result["records"][0]), {"id", "Name", "Area"})
        self.assertEqual(result["records"][0]["Name"], "Fire Station 3")
        self.assertEqual(result["records"][0]["Area"], 1118.446)
        with self.assertRaises(ValueError):
            query(self.catalog, "Fire Station", "sum", field="Name")
        with self.assertRaises(ValueError):
            query(self.catalog, "Fire Station", "list", fields=["invented_field"])


if __name__ == "__main__":
    unittest.main()
