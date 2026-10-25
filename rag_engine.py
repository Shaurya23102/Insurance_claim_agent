"""
Insurance Claim RAG Engine
--------------------------
Parent-child retrieval with HNSW indexing, metadata filtering,
dual-tier memory, query routing & rewriting.

Designed for deployment: no local paths, ephemeral ChromaDB,
model pulled from Hugging Face Hub.
"""
#$env:GROQ_API_KEY = "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"
#D:\AIAGENTS\venv\Scripts\python.exe -m streamlit run D:\AIAGENTS\streamlit_deploy\app.py

import os
import json
import re
import uuid
import numpy as np
import redis
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# ─── Configuration ───────────────────────────────────────────
HF_MODEL_NAME = "shaurya23102/insurance-embedding-model"
POLICY_FILENAME = "insurance_policy.md"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


# ─── Embedding Model (singleton) ────────────────────────────

EMBEDDING_MODEL_NAME = "shaurya23102/insurance-embedding-model"

_model_cache = None

def get_embedding_model():
    global _model_cache
    if _model_cache is None:
        print("[*] Downloading embedding model from HF Hub...")
        _model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("[OK] Embedding model loaded.")
    return _model_cache


def encode_texts(texts):
    """Encode a list of strings into L2-normalised embeddings."""
    model = get_embedding_model()
    if isinstance(texts, str):
        texts = [texts]
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


# ─── Document Parsing ────────────────────────────────────────
def _extract_clause(section_title):
    match = re.match(r'^\s*([a-zA-Z0-9]+)\.?\s*', section_title)
    return match.group(1) if match else "N/A"


def parse_policy(policy_path):
    """Split the policy markdown by ## headings into section dicts."""
    with open(policy_path, "r", encoding="utf-8") as f:
        content = f.read()

    parts = re.split(r'^(##\s+.*)$', content, flags=re.MULTILINE)

    sections = []
    first_part = parts[0].strip()
    if first_part:
        sections.append({"title": "Preamble", "content": first_part})

    for i in range(1, len(parts), 2):
        title = parts[i].replace("##", "").strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections.append({"title": title, "content": body})

    return sections


# ─── Index Builder ───────────────────────────────────────────
def build_index(policy_path):
    """
    Parse the policy → parent/child chunks → embed children →
    store in Chroma HNSW collection.

    Returns (chroma_client, policy_collection, memory_collection, parent_docstore, section_titles)
    """
    sections = parse_policy(policy_path)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500, chunk_overlap=250
    )

    parent_docstore = {}
    child_documents, child_metadatas, child_ids = [], [], []

    for sec in sections:
        sec_title = sec["title"]
        clause = _extract_clause(sec_title)

        parents = parent_splitter.split_text(sec["content"])
        for parent_text in parents:
            parent_id = str(uuid.uuid4())
            parent_docstore[parent_id] = {
                "text": parent_text,
                "section_title": sec_title,
                "clause_number": clause,
            }

            children = child_splitter.split_text(parent_text)
            for child_text in children:
                child_id = str(uuid.uuid4())
                child_documents.append(child_text)
                child_ids.append(child_id)
                child_metadatas.append({
                    "parent_id": parent_id,
                    "policy_name": POLICY_FILENAME,
                    "section_title": sec_title,
                    "subsection_title": sec_title,
                    "clause_number": clause,
                })

    # Persistent Chroma client (saves to disk, survives restarts)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_path = os.path.join(script_dir, "chroma_db")
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    policy_collection = chroma_client.get_or_create_collection(
        name="policy_collection",
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 100,
            "hnsw:M": 16,
        },
    )

    # Only index if the collection is empty (skip on subsequent runs)
    if policy_collection.count() == 0:
        print("[*] Indexing chunks into ChromaDB (first run)...")
        batch_size = 100
        for i in range(0, len(child_documents), batch_size):
            batch_docs = child_documents[i : i + batch_size]
            batch_metas = child_metadatas[i : i + batch_size]
            batch_ids = child_ids[i : i + batch_size]
            batch_embs = encode_texts(batch_docs)

            policy_collection.add(
                embeddings=batch_embs,
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids,
            )
        print(f"[OK] Index built -- {len(child_documents)} child chunks, "
              f"{len(parent_docstore)} parent chunks.")
    else:
        print(f"[OK] Loaded existing index from disk -- {policy_collection.count()} chunks already indexed.")

    # Memory collection (for long-term conversational memory)
    memory_collection = chroma_client.get_or_create_collection(
        name="memory_collection",
        metadata={"hnsw:space": "cosine"},
    )

    section_titles = sorted(set(s["title"] for s in sections))

    return chroma_client, policy_collection, memory_collection, parent_docstore, section_titles


# ─── Semantic Cache (Redis) ──────────────────────────────────
class SemanticCache:
    """
    Redis-backed semantic cache. If a new query's embedding has
    cosine similarity >= threshold with any cached query, return
    the cached response instead of running the full RAG pipeline.
    """

    CACHE_PREFIX = "sem_cache:"
    COUNTER_KEY = "sem_cache:counter"

    def __init__(self, redis_url="redis://localhost:6379/0", threshold=0.85):
        self.threshold = threshold
        self.client = redis.from_url(redis_url, decode_responses=False)
        try:
            self.client.ping()
            self.connected = True
            print(f"[OK] Redis semantic cache connected (threshold={threshold}).")
        except redis.ConnectionError:
            self.connected = False
            print("[WARN] Redis not available -- semantic cache disabled.")

    def _cosine_similarity(self, a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    def lookup(self, query_embedding):
        """
        Check all cached embeddings for a cosine similarity hit.
        Returns (hit: bool, cached_data: dict | None, similarity: float)
        """
        if not self.connected:
            return False, None, 0.0

        count = self.client.get(self.COUNTER_KEY)
        if not count:
            return False, None, 0.0

        best_sim = 0.0
        best_data = None

        for i in range(int(count)):
            raw = self.client.get(f"{self.CACHE_PREFIX}{i}")
            if not raw:
                continue
            entry = json.loads(raw)
            sim = self._cosine_similarity(query_embedding, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_data = entry

        if best_sim >= self.threshold and best_data:
            return True, best_data, best_sim

        return False, None, best_sim

    def store(self, query_text, query_embedding, response, citations, routing_info):
        """Store a query + response in the cache."""
        if not self.connected:
            return

        idx = self.client.get(self.COUNTER_KEY)
        idx = int(idx) if idx else 0

        entry = {
            "query": query_text,
            "embedding": query_embedding,
            "response": response,
            "citations": citations,
            "routing_info": routing_info,
        }
        self.client.set(f"{self.CACHE_PREFIX}{idx}", json.dumps(entry))
        self.client.set(self.COUNTER_KEY, str(idx + 1))

    def flush(self):
        """Clear all cached entries."""
        if not self.connected:
            return
        count = self.client.get(self.COUNTER_KEY)
        if count:
            for i in range(int(count)):
                self.client.delete(f"{self.CACHE_PREFIX}{i}")
        self.client.delete(self.COUNTER_KEY)
        print("[OK] Semantic cache flushed.")

    def size(self):
        if not self.connected:
            return 0
        count = self.client.get(self.COUNTER_KEY)
        return int(count) if count else 0


# ─── Agent Class ─────────────────────────────────────────────
class InsuranceClaimAgent:
    def __init__(self, policy_collection, memory_collection, parent_docstore,
                 semantic_cache=None):
        self.short_term_memory = []
        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=GROQ_API_KEY,
            temperature=0,
        )
        self.policy_collection = policy_collection
        self.memory_collection = memory_collection
        self.parent_docstore = parent_docstore
        self.semantic_cache = semantic_cache

    # ── Memory ──────────────────────────────────────────────
    def store_turn(self, user_query, agent_response):
        if len(self.short_term_memory) >= 4:  # 2 turns × 2 entries
            oldest_user = self.short_term_memory.pop(0)
            oldest_agent = self.short_term_memory.pop(0)
            mem_text = f"User: {oldest_user['content']}\nAgent: {oldest_agent['content']}"
            mem_emb = encode_texts([mem_text])[0]
            self.memory_collection.add(
                embeddings=[mem_emb],
                documents=[mem_text],
                ids=[str(uuid.uuid4())],
            )
        self.short_term_memory.append({"role": "user", "content": user_query})
        self.short_term_memory.append({"role": "assistant", "content": agent_response})

    # ── Query Routing ───────────────────────────────────────
    def classify_query(self, query):
        if not self.short_term_memory:
            return "Type A"

        sys_msg = (
            "You are a router classifier. Classify the user query:\n"
            "- 'Type A': Self-contained, no references to prior turns.\n"
            "- 'Type B': Context-dependent, uses pronouns or references.\n"
            "Respond ONLY with 'Type A' or 'Type B'."
        )
        buf = "\n".join(
            f"{t['role']}: {t['content']}" for t in self.short_term_memory
        )
        resp = self.llm.invoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=f"Buffer:\n{buf}\n\nQuery: {query}"),
        ])
        cls = resp.content.strip()
        return cls if cls in ("Type A", "Type B") else "Type A"

    # ── Query Rewriting ─────────────────────────────────────
    def rewrite_query(self, query):
        sys_msg = (
            "Rewrite this context-dependent query into a fully self-contained "
            "question using the conversational buffer. "
            "Return ONLY the rewritten question."
        )
        buf = "\n".join(
            f"{t['role']}: {t['content']}" for t in self.short_term_memory
        )
        resp = self.llm.invoke([
            SystemMessage(content=sys_msg),
            HumanMessage(content=f"Buffer:\n{buf}\n\nAmbiguous Query: {query}"),
        ])
        return resp.content.strip()

    # ── Long-Term Memory Retrieval ──────────────────────────
    def get_long_term_memory(self, query):
        if self.memory_collection.count() == 0:
            return ""
        qe = encode_texts([query])[0]
        res = self.memory_collection.query(
            query_embeddings=[qe],
            n_results=min(2, self.memory_collection.count()),
        )
        if res and res["documents"] and res["documents"][0]:
            return "\n\n".join(res["documents"][0])
        return ""

    # ── Policy Retrieval ────────────────────────────────────
    def retrieve_policy_context(self, query, metadata_filter=None):
        qe = encode_texts([query])[0]

        query_kwargs = {"query_embeddings": [qe], "n_results": 3}
        if metadata_filter:
            query_kwargs["where"] = metadata_filter

        results = self.policy_collection.query(**query_kwargs)

        parents = []
        citations_text = ""

        if results and results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                meta = results["metadatas"][0][idx]
                doc = results["documents"][0][idx]
                dist = results["distances"][0][idx] if "distances" in results else 0.0
                score = 1.0 - dist

                parent_data = self.parent_docstore.get(meta["parent_id"], {})
                parent_text = parent_data.get("text", "")

                parents.append({"text": parent_text, "metadata": meta})
                citations_text += (
                    f"━━━ CHUNK {idx + 1}  (Similarity: {score:.4f}) ━━━\n"
                    f"📑 Section : {meta['section_title']}\n"
                    f"📌 Clause  : {meta['clause_number']}\n"
                    f"📄 Policy  : {meta['policy_name']}\n\n"
                    f"── Child Snippet ──\n{doc}\n\n"
                    f"── Full Parent Section ──\n{parent_text}\n"
                    f"{'═' * 60}\n\n"
                )

        return parents, citations_text

    # ── Full Pipeline ───────────────────────────────────────
    def generate_response(self, user_query, metadata_filter=None):
        original = user_query
        routing_info = {}

        # 0. Check Semantic Cache (Redis)
        query_emb = encode_texts([user_query])[0]
        if self.semantic_cache and metadata_filter is None:
            hit, cached_data, similarity = self.semantic_cache.lookup(query_emb)
            if hit and cached_data:
                print(f"[CACHE HIT] Sim={similarity:.4f} | Matched: '{cached_data.get('query')}'")
                cached_routing = dict(cached_data.get("routing_info", {}))
                cached_routing["cache_hit"] = True
                cached_routing["cache_similarity"] = round(similarity, 4)
                cached_routing["matched_query"] = cached_data.get("query", "")

                # Maintain conversational memory continuity
                self.store_turn(original, cached_data["response"])
                return cached_data["response"], cached_data.get("citations", ""), cached_routing

        # 1. Routing
        q_type = self.classify_query(user_query)
        routing_info["type"] = q_type
        routing_info["cache_hit"] = False

        if q_type == "Type B":
            rewritten = self.rewrite_query(user_query)
            routing_info["rewritten_query"] = rewritten
            user_query = rewritten

        # 2. Retrieval
        policy_results, citations_text = self.retrieve_policy_context(
            user_query, metadata_filter
        )
        lt_mem = self.get_long_term_memory(user_query)

        # 3. Assemble context
        ctx = "\n\n".join(
            f"Section: {r['metadata']['section_title']} | "
            f"Clause: {r['metadata']['clause_number']}\n{r['text']}"
            for r in policy_results
        )

        sys_prompt = (
            "You are an intelligent Insurance Claim AI Agent. "
            "Answer ONLY from the retrieved policy chunks.\n"
            "Cite specific sections and clause numbers "
            "(e.g. 'As per Section 4.2 – Exclusions').\n"
            "Distinguish covered vs excluded. "
            "Flag ambiguous cases for human escalation.\n"
            "Tone: concise, professional, empathetic."
        )

        user_prompt = f"Retrieved Policy Context:\n{ctx}\n\n"
        if lt_mem:
            user_prompt += f"Long-Term Memory:\n{lt_mem}\n\n"
        if self.short_term_memory:
            buf = "\n".join(
                f"{t['role']}: {t['content']}" for t in self.short_term_memory
            )
            user_prompt += f"Short-Term Buffer:\n{buf}\n\n"
        user_prompt += f"Question: {user_query}"

        # 4. LLM call
        resp = self.llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt),
        ])

        # 5. Save to memory
        self.store_turn(original, resp.content)

        # 6. Save to Semantic Cache if applicable
        if self.semantic_cache and metadata_filter is None:
            self.semantic_cache.store(
                query_text=original,
                query_embedding=query_emb,
                response=resp.content,
                citations=citations_text,
                routing_info=routing_info,
            )

        return resp.content, citations_text, routing_info
