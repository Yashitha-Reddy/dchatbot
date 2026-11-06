import os
import unittest
from unittest.mock import Mock, patch
from geo_chat import GeoPlan, respond
from postgres_store import feature_locations


CATALOG = {"layers": [{"name": "Fire Station", "source": "Buildings/Fire_Station.geojson",
                       "labels": [], "fields": ["Name", "Area_Type"]}]}


class LocationChecks(unittest.TestCase):
    @patch("postgres_store.connect")
    def test_locations_are_derived_from_geometry_with_parameterized_filters(self, connect):
        conn = connect.return_value.__enter__.return_value
        conn.execute.return_value.fetchone.return_value = (1, 0)
        conn.execute.return_value.fetchall.return_value = [
            ("id", {"Name": "Fire Station 3", "Area_Type": "Airside"}, 77.0988, 28.5665)]
        result = feature_locations(CATALOG, "Fire Station", name="Fire Station 3")
        self.assertEqual(result["records"][0]["latitude"], 28.5665)
        self.assertEqual(result["records"][0]["longitude"], 77.0988)
        sql, params = conn.execute.call_args.args
        self.assertIn("ST_PointOnSurface", sql)
        self.assertNotIn("Fire Station 3", sql)
        self.assertIn("Fire Station 3", params)

    @patch("postgres_store.feature_locations")
    @patch("geo_chat.structured_query")
    def test_where_question_displays_name_and_coordinates(self, planner, locations):
        planner.return_value = GeoPlan(operation="locations", layer="Fire Station")
        locations.return_value = {"records": [{"id": "id", "name": "Fire Station 3",
            "area_type": "Airside", "latitude": 28.5665, "longitude": 77.0988}],
            "truncated": False, "excluded_invalid_geometries": 0,
            "source": "Buildings/Fire_Station.geojson"}
        store = Mock()
        store.similarity_search.return_value = []
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://unused"}):
            answer = respond("In what locations are fire stations?", [], CATALOG, store, Mock())
        self.assertIn("Fire Station 3 (Airside): latitude 28.5665, longitude 77.0988", answer)
        self.assertIn("not verified entrances", answer)
