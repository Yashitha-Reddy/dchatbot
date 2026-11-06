import os
import unittest
from unittest.mock import patch

from geo_data import query
from postgres_store import count_records


INFO = {"source": "Other_Data/Trees.geojson", "name": "Trees", "labels": [],
        "fields": ["Name", "Area_Type", "Ground_Elv"], "numeric_fields": ["Ground_Elv"]}


class PostgresCountChecks(unittest.TestCase):
    @patch("postgres_store.connect")
    def test_count_uses_aggregate_without_sort_or_fetchall(self, connect):
        conn = connect.return_value.__enter__.return_value
        conn.execute.return_value.fetchone.return_value = (12979, 0)
        result = count_records(INFO)
        self.assertEqual(result["value"], 12979)
        sql, params = conn.execute.call_args.args
        self.assertIn("count(*)", sql)
        self.assertNotIn("ORDER BY", sql)
        conn.execute.return_value.fetchall.assert_not_called()
        self.assertEqual(params, [INFO["source"]])

    @patch("postgres_store.connect")
    def test_bounds_and_scope_are_parameterized(self, connect):
        conn = connect.return_value.__enter__.return_value
        conn.execute.return_value.fetchone.return_value = (2, 1)
        result = count_records(INFO, field="Ground_Elv", name="test'name", area_type="Airside", minimum=200, maximum=230)
        sql, params = conn.execute.call_args.args
        self.assertNotIn("test'name", sql)
        self.assertIn("test'name", params)
        self.assertEqual(params[-2:], [200, 230])
        self.assertEqual(result["missing_measurements"], 1)

    @patch("postgres_store.records", side_effect=AssertionError("Do not fetch all rows for a count"))
    @patch("postgres_store.count_records")
    def test_query_dispatches_count_to_database(self, count, records):
        count.return_value = {"value": 12979}
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://unused"}):
            self.assertEqual(query({"layers": [INFO]}, "Trees", "count")["value"], 12979)
        count.assert_called_once()
