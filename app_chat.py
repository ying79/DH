# app_chat.py — Simple Kanji Chat UI for Aozora RAG

import streamlit as st
from rag_core import RagPipeline
import traceback

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
# Chat state
# ------------------------------------------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {role, content}

def add_msg(role: str, content: str):
    st.session_state.chat.append({"role": role, "content": content})

def render_answer(ans: str):
    """
    Show main text and Sources block nicely.
    We split on 'Sources:' and render each source line separately.
    """
    if "Sources:" not in ans:
        st.markdown(ans)
        return

    body, sources = ans.split("Sources:", 1)
    st.markdown(body.strip())

    st.markdown("**Sources:**")
    for line in sources.splitlines():
        line = line.strip()
        if not line:
            continue
        # 每一行單獨顯示，就不會被合併成一大段
        st.markdown(line)

# Show history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        # 過去訊息也用同樣的渲染邏輯，這樣 Sources 一致
        if m["role"] == "assistant":
            render_answer(m["content"])
        else:
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

    # user message
    add_msg("user", q)
    with st.chat_message("user"):
        st.markdown(q)

    # assistant message
    with st.chat_message("assistant"):
        try:
            with st.spinner("Looking it up in Aozora Bunko…"):
                ans = rag.answer_unified(q, top_k=10)
        except Exception as e:
            st.error(f"Error while generating answer: {type(e).__name__}: {repr(e)}")
            st.code(traceback.format_exc())

        render_answer(ans)

    add_msg("assistant", ans)
    st.rerun()
