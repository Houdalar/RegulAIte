import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv

from app.ingest import load_and_split
from app.embed_store import build_or_load_store
from app.qa_chain import create_qa_chain

load_dotenv()

st.set_page_config(page_title="RegulAIte", layout="wide")

# ─── Custom styling ─────────────────────────────────────────────
st.markdown(
    """
    <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css\">
    <style>
        .stButton>button {
            background-color: #28a745;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Session State Init ─────────────────────────────────────────
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "qa" not in st.session_state:
    st.session_state.qa = None
if "history" not in st.session_state:
    st.session_state.history = []
if "provider" not in st.session_state:
    st.session_state.provider = "openai"

# Model switcher
use_qwen = st.toggle("Use Qwen model", value=st.session_state.provider == "qwen")
st.session_state.provider = "qwen" if use_qwen else "openai"

# ─── 1. Analyze Documents ───────────────────────────────────────
if not st.session_state.indexed:
    st.header("1. Analyze Documents")
    uploaded = st.file_uploader("Upload PDF/TXT to analyze", accept_multiple_files=True)
    if st.button("Start Analysis") and uploaded:
        with st.spinner("Indexing documents..."):
            os.makedirs("data/raw", exist_ok=True)
            for f in uploaded:
                with open(f"data/raw/{f.name}", "wb") as out:
                    out.write(f.read())
            chunks = load_and_split("data/raw")
            build_or_load_store(chunks, persist_dir="vectorstore")
            st.session_state.qa = create_qa_chain(
                persist_dir="vectorstore", provider=st.session_state.provider
            )
            st.session_state.indexed = True

            sug_res = st.session_state.qa({"question": "List the main policy topics covered in these documents.", "chat_history": []})
            topics = [
                line.strip("• ").strip()
                for line in (sug_res.get("answer") or "").split("\n")
                if line.strip()
            ]
            short = ", ".join(topics[:3])
            welcome = (
                f"<i class='fas fa-robot'></i> Hi, I'm <b>RegulAIte</b>. "
                f"How can I assist you today? You can ask about topics like {short}."
            )
            st.session_state.history.append(("assistant", welcome))

        st.success("Analysis complete! You can start asking questions below.")

# ─── 2. Chat Interface ──────────────────────────────────────────
if st.session_state.indexed:
    st.header("2. Ask Your Policy Questions")

    if st.button("Clear Chat"):
        st.session_state.history = []

    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.markdown(msg, unsafe_allow_html=True)

    user_input = st.chat_input("Type your question here…")
    if user_input:
        normalized = user_input.strip().lower()
        if normalized in ("hi", "hello", "hey"):
            reply = (
                "Hello! I’m RegulAIte. How can I assist you today? "
                "You can ask about policies like sick leave, pay raise, etc."
            )
            st.session_state.history.append(("assistant", reply))
            st.chat_message("assistant").markdown(reply, unsafe_allow_html=True)
        else:
            st.session_state.history.append(("user", user_input))
            st.chat_message("user").markdown(user_input)

            history_pairs = []
            flat = st.session_state.history
            for i in range(0, len(flat) - 1, 2):
                if flat[i][0] == "user" and flat[i + 1][0] == "assistant":
                    history_pairs.append((flat[i][1], flat[i + 1][1]))

            res = st.session_state.qa({"question": user_input, "chat_history": history_pairs})
            answer = res["answer"]

            st.session_state.history.append(("assistant", answer))
            st.chat_message("assistant").markdown(answer, unsafe_allow_html=True)

            with st.expander("Sources", expanded=False):
                unique = []
                seen = set()
                for doc in res["source_documents"]:
                    snippet = doc.page_content.strip().replace("\n", " ")
                    if snippet not in seen:
                        seen.add(snippet)
                        unique.append((doc.metadata.get("source", "unknown"), snippet))
                for src, snippet in unique:
                    st.markdown(f"- **{src}**: {snippet[:300]}…")

