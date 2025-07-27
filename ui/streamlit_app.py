import os, sys
import streamlit as st
from dotenv import load_dotenv
from streamlit.runtime.scriptrunner.script_runner import RerunException

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.ingest import load_and_split
from app.embed_store import build_or_load_store
from app.qa_chain import create_qa_chain

# ─── Config ─────────────────────────────────────────────────────
load_dotenv()
st.set_page_config(page_title="RegulAIte", layout="centered")

# Minimal button styling (Back transparent, Finish green)
st.markdown(
    """
    <style>
    div[data-testid="stButton"][data-key="back_btn"] button{
        background: transparent !important; color:#333 !important; border:1px solid #ccc !important;
    }
    div[data-testid="stButton"][data-key="finish_btn"] button{
        background:#28a745 !important; color:#fff !important; border:1px solid #28a745 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── Helpers ────────────────────────────────────────────────────
def show_sources(md_container, docs):
    """Render unique sources inside an expander."""
    seen = set()
    lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page") or d.metadata.get("loc", "")
        snippet = d.page_content.strip().replace("\n", " ")
        key = (src, page, snippet[:120])
        if key in seen:
            continue
        seen.add(key)
        short = snippet[:240] + ("…" if len(snippet) > 240 else "")
        lines.append(f"**{len(lines)+1}. {src}**{(' · p.'+str(page)) if page else ''}: “{short}”")
    if lines:
        with md_container.expander("Sources"):
            for line in lines:
                st.markdown(line)

def build_history_pairs(history):
    """Return list[(user_text, assistant_text)] from history tuples (role, text)."""
    pairs = []
    # iterate two by two
    for (r1, t1), (r2, t2) in zip(history[::2], history[1::2]):
        if r1 == "user" and r2 == "assistant":
            pairs.append((t1, t2))
    return pairs

MAX_WORDS = 50

# ─── State ──────────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.update(
        step=1,
        history=[],
        qa=None,
        indexed=False,
        provider="openai",
    )

# ─── Step 1: Upload ─────────────────────────────────────────────
if st.session_state.step == 1:
    st.header("1) Upload PDF/TXT files")
    files = st.file_uploader("", accept_multiple_files=True)
    if st.button("Next"):
        if not files:
            st.warning("Please upload at least one file.")
        else:
            os.makedirs("data/raw", exist_ok=True)
            for f in files:
                with open(f"data/raw/{f.name}", "wb") as out:
                    out.write(f.read())
            st.session_state.step = 2
            st.rerun()

# ─── Step 2: Model Choice ───────────────────────────────────────
elif st.session_state.step == 2:
    st.header("2) Choose LLM Provider")
    model = st.selectbox("Provider", ["OpenAI", "Qwen"],
                         index=0 if st.session_state.provider == "openai" else 1)

    col1, col2 = st.columns(2)
    if col1.button("Back", key="back_btn"):
        st.session_state.step = 1
        st.rerun()

    if col2.button("Start Chat", key="finish_btn"):
        st.session_state.provider = model.lower()
        st.info("Analyzing your documents, please wait…", icon="🔍")
        with st.spinner("Indexing documents into vector store…"):
            chunks = load_and_split("data/raw")
            build_or_load_store(chunks, persist_dir="vectorstore")
            st.session_state.qa = create_qa_chain(
                persist_dir="vectorstore",
                provider=st.session_state.provider
            )
            st.session_state.indexed = True
            st.session_state.history = [
                ("assistant", "Hi! I’m RegulAIte. Ask me anything about your docs.")
            ]
        st.session_state.step = 3
        st.rerun()

# ─── Step 3: Chat ───────────────────────────────────────────────
elif st.session_state.step == 3 and st.session_state.indexed:
    st.header("3) Chat with RegulAIte")

    # render chat so far
    for role, text in st.session_state.history:
        st.chat_message(role).markdown(text)

    # input
    user_input = st.chat_input("Type your question… (max 50 words)")
    if user_input:
        if len(user_input.split()) > MAX_WORDS:
            st.warning(f"Your question is over {MAX_WORDS} words—please shorten it.")
        else:
            # echo user instantly
            st.session_state.history.append(("user", user_input))
            st.chat_message("user").markdown(user_input)

            # status while answering
            searching = st.empty()
            searching.info("Searching…")

            # assistant container (for streaming + sources)
            assistant_box = st.chat_message("assistant")
            stream_placeholder = assistant_box.empty()

            # build proper chat history pairs
            history_pairs = build_history_pairs(st.session_state.history)

            # call chain
            res = st.session_state.qa({
                "question": user_input,
                "chat_history": history_pairs
            })

            searching.empty()
            answer = res["answer"]
            stream_placeholder.markdown(answer)  # final text
            st.session_state.history.append(("assistant", answer))

            # show sources (deduped)
            show_sources(assistant_box, res.get("source_documents", []))

    # footer buttons (bottom)
    st.write("")  # spacer
    st.write("")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        if st.button("Back", key="back_btn_footer"):
            st.session_state.step = 2
            st.rerun()
    with fcol2:
        if st.button("Finish", key="finish_btn_footer"):
            for k in ("step", "history", "qa", "indexed", "provider"):
                st.session_state.pop(k, None)
            st.rerun()
