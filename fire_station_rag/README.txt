GeoJSON RAG chatbot — NVIDIA Nemotron + LangChain

The only LLM provider is NVIDIA. See DEPLOYMENT.md for setup.

From fire_station_rag:
  conda activate geo-rag
  python -m pip install -r requirements.txt
  python -m uvicorn api:app --host 127.0.0.1 --port 8000 --workers 1

The application loads .env automatically. Set NVIDIA_API_KEY privately and use
LLM_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b.

Browser/API testing: http://127.0.0.1:8000/docs
All-layer terminal chat: python geo_chat.py

The model interprets questions; validated queries calculate results from the
complete selected layer. Numeric units and vertical datum remain unverified.
