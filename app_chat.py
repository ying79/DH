# app_chat.py — Simple Kanji Chat UI for Aozora RAG

import streamlit as st
from rag_core import RagPipeline

# ------------------------------------------------------------
# Page layout & header
# ------------------------------------------------------------
st.set_page_config(
    page_title="Japanese Literary Name-Kanji Context Finder",
    page_icon="🏯",
    layout="wide",
)

st.markdown(
    """
    <h1 style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.2rem;">
      <span>🏯</span>
      <span>Japanese Literary Name-Kanji Context Finder</span>
    </h1>
    <p style="font-size:1.05rem; color:#555;">
      A simple chat interface for kanji lookup with Aozora Bunko.
    </p>
    <hr style="margin-top:0.6rem; margin-bottom:0.8rem;">
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# RAG pipeline (cached)
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_pipeline():
    # Uses default config.yaml / data directory
    return RagPipeline(config_path="config.yaml", data_dir="data")

rag = get_pipeline()

# ------------------------------------------------------------
# Simple chat state
# ------------------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {role, content}

def add_msg(role: str, content: str):
    st.session_state.chat.append({"role": role, "content": content})

# Show history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ------------------------------------------------------------
# Chat input
# ------------------------------------------------------------
prompt = st.chat_input(
    "Enter a kanji to look up (e.g., 瑩, 祺)"
)

if prompt:
    q = prompt.strip()
    if not q:
        st.stop()

    # show user message
    add_msg("user", q)
    with st.chat_message("user"):
        st.markdown(q)

    # assistant response
    with st.chat_message("assistant"):
        try:
            with st.spinner("Looking it up in Aozora Bunko…"):
                # unified answer: if it's a single kanji → char_dialog,
                # otherwise → normal RAG answer
                ans = rag.answer_unified(q, top_k=10)
        except Exception as e:
            ans = f"(Error while generating answer: {e})"
            st.error(ans)

        st.markdown(ans)

    add_msg("assistant", ans)
    st.rerun()
