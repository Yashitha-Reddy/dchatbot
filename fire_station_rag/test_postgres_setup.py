import unittest
from unittest.mock import Mock
from postgres_store import check_extensions


class ExtensionChecks(unittest.TestCase):
    def test_missing_postgis_is_reported(self):
        connection = Mock()
        connection.execute.return_value.fetchall.return_value = []
        with self.assertRaisesRegex(RuntimeError, "postgis"):
            check_extensions(connection)

    def test_postgis_alone_is_sufficient(self):
        connection = Mock()
        connection.execute.return_value.fetchall.return_value = [("postgis",)]
        check_extensions(connection)
