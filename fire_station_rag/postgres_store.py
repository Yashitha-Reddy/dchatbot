"""Parameterized PostGIS operations. Never execute SQL supplied by the model."""
import json
import math
import os
import sqlite3


def connect():
    import psycopg
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise ValueError("Set DATABASE_URL in the private .env file before using PostgreSQL")
    return psycopg.connect(url, connect_timeout=10, options="-c statement_timeout=30000")


def check_extensions(conn):
    available = {r[0] for r in conn.execute(
        "SELECT name FROM pg_available_extensions WHERE name='postgis'").fetchall()}
    missing = {"postgis"} - available
    if missing:
        raise RuntimeError("Install PostgreSQL server extensions first: " + ", ".join(sorted(missing))
                           + ". See POSTGRES_SETUP.md; pip cannot install server extensions.")


def migrate(catalog, source_db):
    if os.getenv("GEO_SOURCE_SRID") != "4326":
        raise ValueError("Confirm GEO_SOURCE_SRID=4326 for CRS84 longitude/latitude sources")
    fingerprint = json.dumps(catalog["signature"])
    with connect() as conn:
        check_extensions(conn)
        conn.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        conn.execute("CREATE TABLE IF NOT EXISTS geo_metadata (id int PRIMARY KEY, signature text)")
        conn.execute("CREATE TABLE IF NOT EXISTS geo_features (id text PRIMARY KEY, layer text NOT NULL, properties jsonb NOT NULL, original_geometry jsonb, geom geometry(Geometry,4326))")
        conn.execute("CREATE INDEX IF NOT EXISTS geo_layer_idx ON geo_features(layer)")
        conn.execute("CREATE INDEX IF NOT EXISTS geo_geog_idx ON geo_features USING gist ((geom::geography))")
        previous = conn.execute("SELECT signature FROM geo_metadata WHERE id=1").fetchone()
        if previous and previous[0] == fingerprint:
            return
        conn.execute("DELETE FROM geo_features")
        source = sqlite3.connect(source_db)
        try:
            with conn.cursor() as cursor:
                cursor.executemany("""INSERT INTO geo_features
                    SELECT %s,%s,%s::jsonb,%s::jsonb,
                    CASE WHEN ST_IsValid(g) AND NOT ST_IsEmpty(g) THEN g ELSE NULL END
                    FROM (SELECT CASE WHEN %s::jsonb IS NULL OR %s::jsonb='null'::jsonb THEN NULL
                          ELSE ST_Force2D(ST_SetSRID(ST_GeomFromGeoJSON(%s),4326)) END g) s""",
                    ((i, layer, props, geom, geom, geom, geom) for i, layer, props, geom in
                     source.execute("SELECT id,layer,properties,geometry FROM features")))
        finally:
            source.close()
        conn.execute("INSERT INTO geo_metadata VALUES (1,%s) ON CONFLICT(id) DO UPDATE SET signature=excluded.signature", (fingerprint,))
        conn.execute("ANALYZE geo_features")


def records(layer):
    with connect() as conn:
        rows = conn.execute("SELECT id,properties FROM geo_features WHERE layer=%s ORDER BY id", (layer,)).fetchall()
    return [(i, json.dumps(props)) for i, props in rows]


def feature_locations(catalog, layer, name=None, area_type=None, limit=10):
    """Return an on-geometry representative point; do not invent street addresses."""
    from geo_data import resolve_layer
    info = resolve_layer(layer, catalog)
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("Location result limit must be between 1 and 50")
    clauses, params = ["layer=%s"], [info["source"]]
    for field, value in (("Name", name), ("Area_Type", area_type)):
        if value is not None:
            if field not in info["fields"]:
                raise ValueError(f"This layer has no {field} field")
            clauses.append("lower(properties->>%s)=lower(%s)")
            params.extend([field, value])
    predicate = " AND ".join(clauses)
    with connect() as conn:
        count, excluded = conn.execute(
            "SELECT count(*),count(*) FILTER (WHERE geom IS NULL) FROM geo_features WHERE " + predicate,
            params).fetchone()
        rows = conn.execute("""SELECT id,properties,
            ST_X(ST_PointOnSurface(geom)),ST_Y(ST_PointOnSurface(geom))
            FROM geo_features WHERE """ + predicate +
            " AND geom IS NOT NULL ORDER BY properties->>'Name',id LIMIT %s",
            [*params, limit + 1]).fetchall()
    return {"source": info["source"], "matched_count": count,
            "excluded_invalid_geometries": excluded, "truncated": len(rows) > limit,
            "records": [{"id": i, "name": p.get("Name"), "area_type": p.get("Area_Type"),
                         "longitude": round(lon, 7), "latitude": round(lat, 7)}
                        for i, p, lon, lat in rows[:limit]],
            "coordinate_system": "WGS84 longitude/latitude (EPSG:4326)",
            "method": "ST_PointOnSurface: representative point on the geometry; not an entrance or address"}


def spatial_query(catalog, layer, operation, longitude=None, latitude=None,
                  radius_m=None, reference_layer=None, reference_name=None, limit=10):
    from geo_data import resolve_layer
    if operation not in {"nearest", "within_radius"}:
        raise ValueError("Unsupported spatial operation")
    if operation == "nearest" and radius_m is not None:
        raise ValueError("Use within_radius when requesting a distance boundary")
    if not isinstance(limit, int) or not 1 <= limit <= 50:
        raise ValueError("Spatial result limit must be between 1 and 50")
    if operation == "within_radius" and (radius_m is None or not math.isfinite(radius_m) or not 0 < radius_m <= 100000):
        raise ValueError("Supply a radius greater than zero and at most 100000 metres")
    target = resolve_layer(layer, catalog)["source"]
    with connect() as conn:
        if reference_layer or reference_name:
            if not reference_layer or not reference_name or longitude is not None or latitude is not None:
                raise ValueError("Specify either coordinates or an exact reference layer and Name")
            reference = resolve_layer(reference_layer, catalog)["source"]
            rows = conn.execute("SELECT ST_AsEWKT(geom) FROM geo_features WHERE layer=%s AND lower(properties->>'Name')=lower(%s)", (reference, reference_name)).fetchall()
            if len(rows) != 1 or not rows[0][0]:
                raise ValueError("Reference feature is missing, ambiguous, or has invalid geometry")
            anchor = rows[0][0]
        else:
            if (longitude is None or latitude is None or not math.isfinite(longitude)
                    or not math.isfinite(latitude) or not -180 <= longitude <= 180 or not -90 <= latitude <= 90):
                raise ValueError("Supply valid longitude and latitude, or a named reference feature")
            anchor = f"SRID=4326;POINT({longitude} {latitude})"
        excluded = conn.execute("SELECT count(*) FROM geo_features WHERE layer=%s AND geom IS NULL", (target,)).fetchone()[0]
        # Radius predicates use the geography index; all distances are horizontal metres.
        predicate = "AND ST_DWithin(geom::geography, ST_GeomFromEWKT(%s)::geography, %s)" if operation == "within_radius" else ""
        params = [anchor, target]
        if predicate:
            params.extend([anchor, radius_m])
        params.append(limit + 1)
        rows = conn.execute("""SELECT id,properties,
            ST_Distance(geom::geography,ST_GeomFromEWKT(%s)::geography) AS distance_m
            FROM geo_features WHERE layer=%s AND geom IS NOT NULL """ + predicate +
            " ORDER BY distance_m,id LIMIT %s", params).fetchall()
    return {"source": target, "records": [{"id": i, "name": p.get("Name"), "distance_m": round(d, 3)} for i,p,d in rows[:limit]],
            "truncated": len(rows) > limit, "excluded_invalid_geometries": excluded,
            "units": "metres", "method": "2D geodesic distance between geometries; Z ignored"}
