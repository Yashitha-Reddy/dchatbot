"""Check, import and verify the PostgreSQL backend without invoking the LLM."""
import argparse
import os
from settings import load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-data", action="store_true", help="Import workspace features into PostGIS and build local layer embeddings")
    parser.add_argument("--verify", action="store_true", help="Check source counts, spatial queries and vector retrieval")
    args = parser.parse_args()
    load_settings()
    if not os.getenv("DATABASE_URL", "").strip():
        parser.exit(1, "Set DATABASE_URL privately in .env. See POSTGRES_SETUP.md.\n")
    from postgres_store import connect, check_extensions
    try:
        with connect() as conn:
            check_extensions(conn)
            print("PostgreSQL connected; PostGIS server files available. Embeddings use the local index.")
        if not (args.import_data or args.verify):
            return
        from geo_data import ingest, DB, query
        from postgres_store import migrate, spatial_query
        catalog = ingest()
        if args.import_data:
            migrate(catalog, DB)
            from geo_chat import build_catalog_store
            store = build_catalog_store(catalog)
            print(f"Imported {catalog['feature_count']} features and {len(catalog['layers'])} layer documents.")
        if args.verify:
            for layer in catalog["layers"]:
                actual = query(catalog, layer["source"], "count")["value"]
                if actual != layer["feature_count"]:
                    raise RuntimeError("Count mismatch: " + layer["source"])
            with connect() as conn:
                count = conn.execute("SELECT count(*) FROM geo_features").fetchone()[0]
                if count != catalog["feature_count"]:
                    raise RuntimeError("Database totals do not match the source catalog")
                print("Extensions:", conn.execute("SELECT extname,extversion FROM pg_extension WHERE extname='postgis'").fetchall())
            nearest = spatial_query(catalog, "Fire Station", "nearest", longitude=77.09, latitude=28.55, limit=1)
            if not nearest["records"]:
                raise RuntimeError("Nearest-feature verification returned no results")
            print("Nearest fire station:", nearest["records"][0])
            radius = spatial_query(catalog, "Trees", "within_radius", reference_layer="Fire Station", reference_name="Fire Station 3", radius_m=500)
            if any(r["distance_m"] > 500.001 for r in radius["records"]):
                raise RuntimeError("Radius query returned an out-of-range feature")
            if not args.import_data:
                from geo_chat import build_catalog_store
                store = build_catalog_store(catalog)
            hits = store.similarity_search("fire station emergency buildings", k=3)
            if not hits:
                raise RuntimeError("Local vector retrieval returned no documents")
            print("Vector retrieval sources:", [d.metadata["source"] for d in hits])
            print(f"Verified all {len(catalog['layers'])} layer counts, geometry distance queries and local vector retrieval.")
    except Exception as exc:
        # Connection exceptions can contain host/user details; never print a DSN.
        if isinstance(exc, (ValueError, RuntimeError)):
            parser.exit(1, str(exc) + "\n")
        parser.exit(1, f"Database setup failed ({type(exc).__name__}). Check credentials, server extensions and database privileges.\n")


if __name__ == "__main__":
    main()
