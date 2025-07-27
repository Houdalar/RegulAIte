import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv

from app.ingest import load_and_split
from app.embed_store import build_or_load_store
from app.qa_chain import create_qa_chain

load_dotenv()

st.set_page_config(page_title="RegulAIte", layout="centered")

# ─── Custom styling ─────────────────────────────────────────────
st.markdown(
    """
    <link rel=\"stylesheet\" href=\"https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css\">
    <style>
        .stButton>button {
            background-color: #28a745;
            color: white;
        }
        .step {
            max-width: 600px;
            margin: auto;
            text-align: center;
            padding-top: 2rem;
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
if "step" not in st.session_state:
    st.session_state.step = 1



# ─── Multi-step Interface ───────────────────────────────────────
if st.session_state.step == 1:
    with st.container():
        st.markdown("<div class='step'>", unsafe_allow_html=True)
        st.header("1. Select a Model")
        choice = st.radio("Choose your language model", ["OpenAI", "Qwen"],
                          index=0 if st.session_state.provider == "openai" else 1)
        if st.button("Next"):
            st.session_state.provider = "openai" if choice == "OpenAI" else "qwen"
            st.session_state.step = 2
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.step == 2:
    with st.container():
        st.markdown("<div class='step'>", unsafe_allow_html=True)
        st.header("2. Analyze Documents")
        uploaded = st.file_uploader("Upload PDF/TXT to analyze", accept_multiple_files=True)
        col1, col2 = st.columns(2)
        back = col1.button("Back")
        start = col2.button("Start Analysis")
        if back:
            st.session_state.step = 1
        if start and uploaded:
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
                st.session_state.history = [
                    ("assistant", "<i class='fas fa-robot'></i> Hi, I'm <b>RegulAIte</b>! How can I assist you today?")
                ]
                st.success("Analysis complete! Start chatting below.")
                st.session_state.step = 3
        st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.step == 3 and st.session_state.indexed:
    with st.container():
        st.markdown("<div class='step'>", unsafe_allow_html=True)
        st.header("3. Ask Your Policy Questions")
        col1, col2 = st.columns(2)
        if col1.button("Back"):
            st.session_state.step = 2
        if col2.button("End Chat"):
            st.session_state.history = []
            st.session_state.qa = None
            st.session_state.indexed = False
            st.session_state.step = 1
            st.experimental_rerun()

        for role, msg in st.session_state.history:
            with st.chat_message(role):
                st.markdown(msg, unsafe_allow_html=True)

        user_input = st.chat_input("Type your question here…")
        if user_input:
            normalized = user_input.strip().lower()
            st.session_state.history.append(("user", user_input))
            st.chat_message("user").markdown(user_input)

            if normalized in ("hi", "hello", "hey"):
                reply = "I'm RegulAIte, your policy assistant. Please ask me about the policies in your uploaded documents."
                st.session_state.history.append(("assistant", reply))
                st.chat_message("assistant").markdown(reply, unsafe_allow_html=True)
            else:
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
        st.markdown("</div>", unsafe_allow_html=True)

