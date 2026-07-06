"""
Insurance Claim RAG Engine
--------------------------
Parent-child retrieval with HNSW indexing, metadata filtering,
dual-tier memory, query routing & rewriting.

Designed for deployment: no local paths, ephemeral ChromaDB,
model pulled from Hugging Face Hub.
"""

import os
import json
import re
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# ─── Configuration ───────────────────────────────────────────────
HF_MODEL_NAME = "shaurya23102/insurance-embedding-model"
POLICY_FILENAME = "insurance_policy.md"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")


# ─── Embedding Model (singleton) ────────────────────────────────
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("⏳ Downloading embedding model from HF Hub...")
        _embedding_model = SentenceTransformer(HF_MODEL_NAME)
        print("✅ Embedding model loaded.")
    return _embedding_model


def encode_texts(texts):
    """Encode a list of strings into L2-normalised embeddings."""
    model = get_embedding_model()
    if isinstance(texts, str):
        texts = [texts]
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


# ─── Document Parsing ────────────────────────────────────────────
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


# ─── Index Builder ───────────────────────────────────────────────
def build_index(policy_path):
    """
    Parse the policy → parent/child chunks → embed children →
    store in Chroma HNSW collection.

    Returns (chroma_client, policy_collection, parent_docstore, section_titles)
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

    # Ephemeral Chroma client (in-memory, no disk needed)
    chroma_client = chromadb.Client()

    policy_collection = chroma_client.create_collection(
        name="policy_collection",
        metadata={
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 100,
            "hnsw:M": 16,
        },
    )

    # Batch-write embeddings
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

    # Memory collection (for long-term conversational memory)
    memory_collection = chroma_client.create_collection(
        name="memory_collection",
        metadata={"hnsw:space": "cosine"},
    )

    section_titles = sorted(set(s["title"] for s in sections))

    print(f"✅ Index built — {len(child_documents)} child chunks, "
          f"{len(parent_docstore)} parent chunks, {len(section_titles)} sections.")

    return chroma_client, policy_collection, memory_collection, parent_docstore, section_titles


# ─── Agent Class ─────────────────────────────────────────────────
class InsuranceClaimAgent:
    def __init__(self, policy_collection, memory_collection, parent_docstore):
        self.short_term_memory = []
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=GROQ_API_KEY,
            temperature=0,
        )
        self.policy_collection = policy_collection
        self.memory_collection = memory_collection
        self.parent_docstore = parent_docstore

    # ── Memory ──────────────────────────────────────────────────
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

    # ── Query Routing ───────────────────────────────────────────
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

    # ── Query Rewriting ─────────────────────────────────────────
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

    # ── Long-Term Memory Retrieval ──────────────────────────────
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

    # ── Policy Retrieval ────────────────────────────────────────
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

    # ── Full Pipeline ───────────────────────────────────────────
    def generate_response(self, user_query, metadata_filter=None):
        original = user_query
        routing_info = {}

        # 1. Routing
        q_type = self.classify_query(user_query)
        routing_info["type"] = q_type

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

        return resp.content, citations_text, routing_info
