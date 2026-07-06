# Insurance Claim AI Agent

A RAG-powered insurance policy assistant with parent-child retrieval,
HNSW indexing, metadata filtering, conversational memory, and query routing.

## Local Development

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Railway Deployment

1. Push this folder to a GitHub repository.
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub Repo**.
3. Select this repo.
4. Add the environment variable:
   - `GROQ_API_KEY` = your Groq API key
5. Railway will auto-detect the `Dockerfile` and deploy.
6. Once deployed, go to **Settings → Networking → Generate Domain** to get your public URL.

## Architecture

- **Embedding Model**: `shaurya23102/insurance-embedding-model` (downloaded from HF Hub on startup)
- **Vector Store**: ChromaDB (ephemeral, rebuilt on each deploy from bundled policy document)
- **LLM**: Llama 3.3 70B via Groq API
- **Memory**: 2-turn short-term buffer + long-term vector store
- **Routing**: Type A (self-contained) / Type B (context-dependent with silent rewriting)
