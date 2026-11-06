# Local PostgreSQL 17 + PostGIS

The chatbot uses PostgreSQL/PostGIS for features and spatial queries. Embeddings
remain in the existing local catalog vector index. No pgvector server extension
or C++ build tools are required.

## 1. Create and enable the database

Open pgAdmin and connect to your local PostgreSQL 17 server. Create a database
named geochat. Select it, open Query Tool and execute:

```sql
CREATE EXTENSION IF NOT EXISTS postgis;
SELECT extname, extversion FROM pg_extension WHERE extname = 'postgis';
```

Use a database owner that can create application tables for this local POC.
Do not reuse a database containing unrelated geo_features tables.

## 2. Configure the private .env

Preserve NVIDIA_API_KEY and your other settings. Set:

```dotenv
DATABASE_URL=postgresql://YOUR_USER:URL_ENCODED_PASSWORD@127.0.0.1:5432/geochat
GEO_SOURCE_SRID=4326
```

URL-encode special characters in username/password values. Keep credentials
private. Confirm source coordinates are CRS84/WGS84 longitude then latitude.
Original Z coordinates remain preserved, but spatial calculations are 2D.

## 3. Import and verify

Stop the API before importing. From fire_station_rag:

```powershell
conda activate geo-rag
python -m pip install -r requirements.txt
python setup_postgres.py
python setup_postgres.py --import-data --verify
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

Import stores all 74,797 source features in geo_features. It preserves properties
and original geometry JSON and builds a GiST geography index. Invalid/empty
geometries remain in original JSON but are excluded from spatial calculations.

The BGE embedding model builds or loads local embeddings for the 83 layer schemas
in data/catalog_vectors.json. Setting DATABASE_URL does not move vectors into
PostgreSQL. Keep that file, its fingerprint, and the embedding model cache persistent.

Verification compares every layer count, executes nearest and radius queries,
and tests local vector retrieval without calling the LLM.

## 4. Test

- Where are the fire stations? (names and representative latitude/longitude)
- Where is Fire Station 3?
- Show locations of cityside fire stations.
- Which fire station is nearest to longitude 77.09 latitude 28.55?
- List trees within 500 metres of Fire Station 3 in Buildings/Fire_Station.geojson.
- How many cityside fire stations are there?
- List fire station names and area.

Nearest/radius queries do not yet combine with attribute filters. Named reference
features must resolve uniquely. Distances are horizontal metres; recorded numeric
attribute units and vertical datum remain unverified. No routing or building-height
inference. Sessions remain SQLite, so use one API worker.

Existing database vector tables, if any, are left untouched and are no longer used.

Location listings use ST_PointOnSurface on stored geometry, not street addresses
or verified entrances. Named localities need separate address data or geocoding.
