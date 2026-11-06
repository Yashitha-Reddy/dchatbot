import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from settings import server_key
from geo_chat import GeoPlan, respond


class DeploymentChecks(unittest.TestCase):
    @patch("postgres_store.spatial_query")
    @patch("geo_chat.structured_query")
    def test_user_coordinates_survive_model_omission(self, planner, spatial):
        planner.return_value = GeoPlan(operation="nearest", layer="Fire Station")
        spatial.return_value = {"records": []}
        store = Mock()
        store.similarity_search.return_value = []
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://unused"}):
            respond("Which fire station is nearest to longitude 77.09 latitude 28.55?", [], {"layers": []}, store, Mock())
        self.assertEqual(spatial.call_args.kwargs["longitude"], 77.09)
        self.assertEqual(spatial.call_args.kwargs["latitude"], 28.55)

    def test_height_cannot_be_substituted_with_ground_elevation(self):
        catalog = {"layers": [{"name": "Fire Station"}]}
        result = respond("What is the height of Fire Station 3?", [], catalog, None, None)
        self.assertIn("height is unavailable", result)
        self.assertIn("ground elevation", result)

    def test_single_aggregate_column_is_validated_as_measurement(self):
        plan = GeoPlan(operation="sum", layer="Fire Station", fields=["Area"])
        self.assertEqual(plan.field, "Area")
        self.assertIsNone(plan.fields)

    def test_generated_key_survives_restart(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            path = Path(folder) / "key"
            with patch.dict(os.environ, {"CHAT_API_KEY": "", "CHAT_KEY_FILE": str(path)}):
                first = server_key()
                self.assertGreaterEqual(len(first), 32)
                self.assertEqual(first, server_key())

    def test_configured_key_takes_precedence(self):
        with patch.dict(os.environ, {"CHAT_API_KEY": "operator-secret"}):
            self.assertEqual(server_key(), "operator-secret")

    @patch("geo_chat.structured_query")
    def test_spatial_without_postgres_is_explicit(self, planner):
        planner.return_value = GeoPlan(operation="nearest", layer="Trees", longitude=77, latitude=28)
        store = Mock()
        store.similarity_search.return_value = []
        with patch.dict(os.environ, {"DATABASE_URL": ""}):
            self.assertIn("require PostgreSQL", respond("nearest trees", [], {"layers": []}, store, Mock()))

    @patch("geo_chat.structured_query")
    def test_spatial_filter_is_not_silently_dropped(self, planner):
        planner.return_value = GeoPlan(operation="nearest", layer="Trees", area_type="Airside")
        store = Mock()
        store.similarity_search.return_value = []
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://unused"}):
            self.assertIn("do not yet support", respond("nearest airside trees", [], {"layers": []}, store, Mock()))

    @patch("postgres_store.connect")
    def test_bad_coordinates_do_not_execute_distance_sql(self, connection):
        from postgres_store import spatial_query
        catalog = {"layers": [{"source": "Trees.geojson", "name": "Trees", "labels": []}]}
        with self.assertRaises(ValueError):
            spatial_query(catalog, "Trees", "nearest", longitude=190, latitude=28)
        connection.return_value.__enter__.return_value.execute.assert_not_called()

    @patch("postgres_store.connect")
    def test_radius_is_required(self, connection):
        from postgres_store import spatial_query
        with self.assertRaises(ValueError):
            spatial_query({"layers": []}, "Trees", "within_radius", longitude=77, latitude=28)
        connection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
