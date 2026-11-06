import os
import unittest
from unittest.mock import Mock, patch

from geo_chat import build_catalog_store


class CatalogBackendChecks(unittest.TestCase):
    def test_database_url_keeps_catalog_vectors_local(self):
        fake_module = Mock()
        catalog = {"layers": [{"source": "Trees.geojson", "name": "Trees"}]}
        path = Mock()
        path.exists.return_value = False
        data = Mock()
        data.__truediv__ = Mock(return_value=path)
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://unused"}), \
             patch.dict("sys.modules", {"langchain_huggingface": fake_module}), \
             patch("geo_chat.DATA", data), \
             patch("geo_chat.InMemoryVectorStore") as vector_store, \
             patch("postgres_store.connect", side_effect=AssertionError("Vector retrieval must stay local")):
            result = build_catalog_store(catalog)
        self.assertIs(result, vector_store.return_value)
        vector_store.return_value.add_documents.assert_called_once()
        vector_store.return_value.dump.assert_called_once()
