# NVIDIA Nemotron setup

## Run locally

From `fire_station_rag`, activate `geo-rag` and install dependencies:

```powershell
conda activate geo-rag
python -m pip install -r requirements.txt
```

For a new installation copy `.env.example` to `.env` only if no `.env` exists.
The API and both CLIs load `.env` automatically; process environment takes precedence.
Set these values privately:

```dotenv
LLM_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
NVIDIA_API_KEY=YOUR_VALID_PROVIDER_KEY
CHAT_API_KEY=
CHAT_MAX_CONCURRENT=2
DATABASE_URL=
GEO_SOURCE_SRID=4326
```

When CHAT_API_KEY is empty, the server creates `data/chat_api_key` once and reuses
it. Display the effective key locally for Swagger authorization:

```powershell
python -c "from settings import load_settings, server_key; load_settings(); print(server_key())"
python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

Open http://127.0.0.1:8000/docs, authorize with the chatbot key, create a session,
and send messages with its returned UUID. The NVIDIA key is separate and stays
on the server. Do not embed either shared secret in a public frontend.

## EC2 without containers

Copy the application and all original GeoJSON folders to the host. Install
Python 3.11, create a virtual environment, then install dependencies:

```sh
cd /path/to/Layer_Wise_Data_GeoJSON/fire_station_rag
python3.11 -m venv .venv
.venv/bin/python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1
```

Configure a private `.env` on the host. Use a service manager for restart on boot.
First startup needs outbound internet for the embedding model download and NVIDIA
requests. Keep `data/` persistent and writable by the service account. The hosted
LLM requires no local GPU. Benchmark RAM, latency and concurrent traffic.

For an internal POC use an SSH tunnel. Before public use configure HTTPS through
a reverse proxy, authenticated backend integration, user-to-session ownership,
request limits and monitoring. Keep port 8000 private. Trial provider access stays
subject to its evaluation terms and quotas.

## Data and optional PostgreSQL

For local PostgreSQL 17 setup, migration and verification see POSTGRES_SETUP.md.

SQLite is the default feature/session store; catalog embeddings are local.
To enable the existing PostgreSQL backend, separately provision PostgreSQL with
PostGIS, then set DATABASE_URL. Embeddings remain in the local vector index.
Confirm source coordinates and
GEO_SOURCE_SRID before importing. PostgreSQL integration still requires validation
on your target host. No database server is installed by this repository.

Nearest/radius spatial queries require PostgreSQL. Ground elevation and coordinate
Z values must not be treated as building height. Recorded units remain unverified.
Counts represent features, including any duplicates present in original files.

Run exactly one API process: session locks are process-local and sessions use
SQLite. Multiple replicas need shared session storage, distributed locking and
per-user authorization. Back up persistent data and test restores. Stop the API
before source updates; restart to ingest and refresh the catalog.

## Verification

```sh
python -m unittest discover -s . -p 'test_*.py'
```

Check /health, /ready, authentication, session isolation, counts, recorded numeric
aggregations, normal chat and unavailable-data questions. Live hosted-model calls
and EC2 deployment need separate verification. No AWS resources are provisioned.
