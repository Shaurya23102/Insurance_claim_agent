"""
Insurance Claim AI Agent — Streamlit Chat Interface
"""

import os
import streamlit as st
from rag_engine import (
    get_embedding_model,
    build_index,
    InsuranceClaimAgent,
    POLICY_FILENAME,
)

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Insurance Claim AI Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    *, .stMarkdown, .stText { font-family: 'Inter', sans-serif !important; }

    /* ── Main area ── */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 900px;
    }

    /* ── Header banner ── */
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

    /* ── Routing badge ── */
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

    /* ── Citation expander ── */
    .stExpander {
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        background: rgba(255,255,255,0.02) !important;
    }

    /* ── Sidebar ── */
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

    /* ── Status pills ── */
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
</style>
""", unsafe_allow_html=True)


# ─── Cached Resource Loading ────────────────────────────────────
@st.cache_resource(show_spinner="🔄 Loading embedding model & building index...")
def load_resources():
    """Load model and build index once — survives Streamlit reruns."""
    _ = get_embedding_model()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    policy_path = os.path.join(script_dir, POLICY_FILENAME)

    chroma_client, policy_col, memory_col, parent_ds, sec_titles = build_index(
        policy_path
    )
    return policy_col, memory_col, parent_ds, sec_titles


policy_col, memory_col, parent_ds, section_titles = load_resources()


# ─── Session State ───────────────────────────────────────────────
if "agent" not in st.session_state:
    st.session_state.agent = InsuranceClaimAgent(
        policy_col, memory_col, parent_ds
    )
if "messages" not in st.session_state:
    st.session_state.messages = []

agent = st.session_state.agent


# ─── Header ──────────────────────────────────────────────────────
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


# ─── Sidebar ─────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/3d-fluency/94/insurance.png",
        width=64,
    )
    st.markdown("### Agent Status")

    mem_count = len(agent.short_term_memory) // 2
    lt_count = agent.memory_collection.count()
    st.markdown(
        f'<span class="status-pill pill-green">🟢 Agent Online</span><br>'
        f'<span class="status-pill pill-blue">💬 Short-term: {mem_count}/2 turns</span><br>'
        f'<span class="status-pill pill-amber">🧠 Long-term: {lt_count} stored</span>',
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
        "1. Your query is **classified** as self-contained (A) or "
        "context-dependent (B).\n"
        "2. Type B queries are **silently rewritten** using conversation "
        "history.\n"
        "3. **Child chunks** (~500 words) are matched via HNSW cosine "
        "search.\n"
        "4. The full **parent section** (~2000 words) is returned as "
        "context.\n"
        "5. The LLM generates a **grounded, cited** answer."
    )

    st.markdown("---")
    if st.button("🗑️  Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.agent = InsuranceClaimAgent(
            policy_col, memory_col, parent_ds
        )
        st.rerun()


# ─── Chat History ────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🛡️"):
        if msg["role"] == "assistant":
            # Show routing badge
            ri = msg.get("routing_info", {})
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

        # Citations expander
        if msg["role"] == "assistant" and msg.get("citations"):
            with st.expander("📎 View Retrieved Chunks (Citations)"):
                st.code(msg["citations"], language="text")


# ─── User Input ──────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your insurance policy…"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(prompt)

    # Build metadata filter
    meta_filter = None
    if filter_section != "All Sections":
        meta_filter = {"section_title": filter_section}

    # Generate response
    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("Searching policy & generating response…"):
            answer, citations, routing_info = agent.generate_response(
                prompt, metadata_filter=meta_filter
            )

        # Routing badge
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

    # Save to session
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "citations": citations,
        "routing_info": routing_info,
    })
