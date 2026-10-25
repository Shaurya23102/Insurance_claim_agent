"""
Insurance Claim AI Agent — Streamlit Chat Interface
"""

import os
import streamlit as st
from rag_engine import (
    get_embedding_model,
    build_index,
    InsuranceClaimAgent,
    SemanticCache,
    POLICY_FILENAME,
)

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Claim AI Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    *, .stMarkdown, .stText { font-family: 'Inter', sans-serif !important; }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    .agent-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .agent-header h1 {
        margin: 0 0 0.3rem 0;
        font-size: 1.6rem;
        font-weight: 700;
        color: #e0f7fa;
        letter-spacing: -0.02em;
    }
    .agent-header p {
        margin: 0;
        font-size: 0.9rem;
        color: #80cbc4;
        line-height: 1.5;
    }

    .route-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        margin-bottom: 0.4rem;
    }
    .route-a { background: #004d40; color: #a7ffeb; }
    .route-b { background: #311b92; color: #d1c4e9; }
    .route-cache { background: #4a148c; color: #f3e5f5; border: 1px solid #ab47bc; }

    .stExpander {
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        background: rgba(255,255,255,0.02) !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #80cbc4;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1.2rem;
    }

    .status-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px 0;
    }
    .pill-green  { background: #1b5e20; color: #a5d6a7; }
    .pill-blue   { background: #0d47a1; color: #90caf9; }
    .pill-amber  { background: #e65100; color: #ffcc80; }
    .pill-purple { background: #4a148c; color: #e1bee7; }
</style>
""", unsafe_allow_html=True)


# ─── Cached Resource Loading ────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading embedding model & building index...")
def load_resources():
    """Load model, index, and semantic cache once — survives Streamlit reruns."""
    _ = get_embedding_model()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    policy_path = os.path.join(script_dir, POLICY_FILENAME)

    chroma_client, policy_col, memory_col, parent_ds, sec_titles = build_index(
        policy_path
    )
    semantic_cache = SemanticCache(threshold=0.85)
    return policy_col, memory_col, parent_ds, sec_titles, semantic_cache


policy_col, memory_col, parent_ds, section_titles, semantic_cache = load_resources()


# ─── Session State ───────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = InsuranceClaimAgent(
        policy_col, memory_col, parent_ds, semantic_cache=semantic_cache
    )
if "messages" not in st.session_state:
    st.session_state.messages = []

agent = st.session_state.agent


# ─── Header ──────────────────────────────────────────────────
st.markdown("""
<div class="agent-header">
    <h1>🛡️ Insurance Claim AI Agent</h1>
    <p>
        Ask about your ICICI Lombard health insurance policy — coverage,
        exclusions, definitions, claim eligibility, and more.<br>
        Every answer is grounded in the actual policy document with cited
        sections and clause numbers.
    </p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/3d-fluency/94/insurance.png",
        width=64,
    )
    st.markdown("### Agent Status")

    mem_count = len(agent.short_term_memory) // 2
    lt_count = agent.memory_collection.count()
    cache_count = semantic_cache.size()
    redis_badge = "🟢 Redis Online" if semantic_cache.connected else "⚪ Redis Offline"
    st.markdown(
        f'<span class="status-pill pill-green">{redis_badge}</span><br>'
        f'<span class="status-pill pill-blue">💬 Short-term: {mem_count}/2 turns</span><br>'
        f'<span class="status-pill pill-amber">🧠 Long-term: {lt_count} stored</span><br>'
        f'<span class="status-pill pill-purple">⚡ Semantic Cache: {cache_count} items (≥0.85)</span>',
        unsafe_allow_html=True,
    )

    st.markdown("### Metadata Filter")
    st.caption("Optionally restrict retrieval to a specific policy section.")
    filter_section = st.selectbox(
        "Section",
        options=["All Sections"] + section_titles,
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        "1. **Semantic Cache Check**: If query similarity $\ge 0.85$, instant response.\n"
        "2. Query is **classified** as Type A or Type B.\n"
        "3. Type B queries are **silently rewritten**.\n"
        "4. **Child chunks** (~500 words) matched via HNSW search.\n"
        "5. Full **parent section** (~2000 words) passed to LLM.\n"
        "6. Answer is stored in **Redis semantic cache** for future queries."
    )

    st.markdown("---")
    if semantic_cache.connected and cache_count > 0:
        if st.button("⚡ Clear Semantic Cache", use_container_width=True):
            semantic_cache.flush()
            st.rerun()

    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent = InsuranceClaimAgent(
            policy_col, memory_col, parent_ds, semantic_cache=semantic_cache
        )
        st.rerun()


# ─── Chat History ────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🛡️"):
        if msg["role"] == "assistant":
            ri = msg.get("routing_info", {})
            if ri.get("cache_hit"):
                sim = ri.get("cache_similarity", 1.0)
                st.markdown(
                    f'<span class="route-badge route-cache">⚡ Cache Hit ({sim:.2f})</span>',
                    unsafe_allow_html=True,
                )
                if "matched_query" in ri:
                    st.caption(f"🎯 Cached from: _{ri['matched_query']}_")
            else:
                badge_cls = "route-a" if ri.get("type") == "Type A" else "route-b"
                badge_label = ri.get("type", "")
                if badge_label:
                    st.markdown(
                        f'<span class="route-badge {badge_cls}">{badge_label}</span>',
                        unsafe_allow_html=True,
                    )
                if "rewritten_query" in ri:
                    st.caption(f"🔄 Rewritten → _{ri['rewritten_query']}_")

        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("📎 View Retrieved Chunks (Citations)"):
                st.code(msg["citations"], language="text")


# ─── User Input ──────────────────────────────────────────────
if prompt := st.chat_input("Ask about your insurance policy…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(prompt)

    meta_filter = None
    if filter_section != "All Sections":
        meta_filter = {"section_title": filter_section}

    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("Searching policy & generating response…"):
            answer, citations, routing_info = agent.generate_response(
                prompt, metadata_filter=meta_filter
            )

        if routing_info.get("cache_hit"):
            sim = routing_info.get("cache_similarity", 1.0)
            st.markdown(
                f'<span class="route-badge route-cache">⚡ Cache Hit ({sim:.2f})</span>',
                unsafe_allow_html=True,
            )
            if "matched_query" in routing_info:
                st.caption(f"🎯 Cached from: _{routing_info['matched_query']}_")
        else:
            badge_cls = "route-a" if routing_info.get("type") == "Type A" else "route-b"
            st.markdown(
                f'<span class="route-badge {badge_cls}">{routing_info.get("type", "")}</span>',
                unsafe_allow_html=True,
            )
            if "rewritten_query" in routing_info:
                st.caption(f"🔄 Rewritten → _{routing_info['rewritten_query']}_")

        st.markdown(answer)

        if citations:
            with st.expander("📎 View Retrieved Chunks (Citations)"):
                st.code(citations, language="text")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "routing_info": routing_info,
    })
