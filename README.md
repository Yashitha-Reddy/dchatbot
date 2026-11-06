# GeoJSON chatbot

LangChain and NVIDIA Nemotron chatbot API for querying local GeoJSON layers.
Supports counts, recorded measurements, feature lists and locations, plus nearest
and radius queries through PostgreSQL/PostGIS. Layer-schema embeddings are stored
in a local vector index; pgvector is not required.

## Setup

1. Use Python 3.11 and install `fire_station_rag/requirements.txt`.
2. Supply your own GeoJSON data in layer folders alongside `fire_station_rag`.
   Source datasets, generated indexes, credentials and chat histories are excluded
   from this repository. Layer names, counts and sample questions in the setup
   documents describe the original development dataset; yours may differ.
3. Copy `fire_station_rag/.env.example` to `fire_station_rag/.env` and configure
   your NVIDIA API key privately. Configure `DATABASE_URL` for PostGIS queries.
4. From `fire_station_rag`, run:

```sh
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open http://127.0.0.1:8000/docs. Authorize with the chatbot key, create a session,
then send messages using its returned UUID. Keep shared server credentials out of
public frontend applications. See the setup documents for key management.

## Documentation

- [Deployment](fire_station_rag/DEPLOYMENT.md)
- [PostgreSQL and PostGIS](fire_station_rag/POSTGRES_SETUP.md)
- [API instructions](fire_station_rag/API_SETUP.txt)

Main modules: `api.py` (HTTP API and sessions), `geo_chat.py` (chat and retrieval),
`geo_data.py` (ingestion and measurements), `postgres_store.py` (spatial queries),
and `providers.py` (NVIDIA integration).

This is a POC with one API worker. Production requires provider access permitted
for production, per-user authorization, HTTPS, rate limits and deployment testing.
Original numeric measurement units and vertical datum must be verified separately.
No Docker setup or local model server is required.
