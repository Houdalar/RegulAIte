import os
import sys

# Ensure your project root is on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv
from app.ingest import load_and_split
from app.embed_store import build_or_load_store
from app.qa_chain import create_qa_chain

load_dotenv()

st.set_page_config(page_title="RegulAIte", layout="wide")

# ─── Global CSS ───────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stTextInput>div>div>input:focus, textarea:focus {
      outline: 2px solid #28a745 !important;
      box-shadow: 0 0 0 0.2rem rgba(40,167,69,.25) !important;
    }
    .stButton>button {
      background-color: #28a745;
      color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Session State Init ─────────────────────────────────────────────
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "qa" not in st.session_state:
    st.session_state.qa = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of (role, msg)
if "suggestions" not in st.session_state:
    st.session_state.suggestions = None

# ─── 1️⃣ Analyze Documents ───────────────────────────────────────────
if not st.session_state.indexed:
    st.header("1️⃣ Analyze Documents")
    uploaded = st.file_uploader(
        "Upload PDF/TXT to analyze", accept_multiple_files=True
    )
    if st.button("Start Analysis") and uploaded:
        with st.spinner("🔍 Indexing documents..."):
            os.makedirs("data/raw", exist_ok=True)
            for f in uploaded:
                with open(f"data/raw/{f.name}", "wb") as out:
                    out.write(f.read())
            chunks = load_and_split("data/raw")
            build_or_load_store(chunks, persist_dir="vectorstore")
            st.session_state.qa = create_qa_chain(persist_dir="vectorstore")
            st.session_state.indexed = True

            # Generate initial topic suggestions
            sug_res = st.session_state.qa({
                "question": "List the main policy topics covered in these documents.",
                "chat_history": []
            })
            st.session_state.suggestions = sug_res["answer"]

        st.success("✅ Analysis complete! Scroll down for suggestions and chat.")

# ─── 2️⃣ Chat Interface ─────────────────────────────────────────────
if st.session_state.indexed:
    st.header("2️⃣ Ask Your Policy Questions")

    # Clear chat resets history but keeps suggestions
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []

    # If it's the first interaction (empty history), show suggestions
    if not st.session_state.history:
        st.markdown("### Try asking about one of these policies:")
        # suggestions is a string like "• Sick Leave\n• Pay Raise\n• Vacation"
        topics = [
            line.strip("• ").strip()
            for line in (st.session_state.suggestions or "").split("\n")
            if line.strip()
        ]
        cols = st.columns(min(len(topics), 3), gap="large")
        for i, topic in enumerate(topics):
            question_text = f"What is the policy regarding {topic.lower()}?"
            with cols[i % len(cols)]:
                st.markdown(f"**{topic}**")
                if st.button(question_text, key=f"sug_{i}"):
                    # inject a formatted question
                    st.session_state.history.append(("user", question_text))

    # Display chat history
    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.write(msg)

    # Chat input
    user_input = st.chat_input("Type your question here…")
    if user_input:
        normalized = user_input.strip().lower()

        # 1️⃣ Greetings
        if normalized in ("hi", "hello", "hey"):
            reply = (
                "Hello! I’m RegulAIte. How can I assist you today? "
                "You can ask about policies like sick leave, pay raise, etc."
            )
            st.session_state.history.append(("assistant", reply))
            st.chat_message("assistant").write(reply)

        # 2️⃣ Otherwise, policy QA
        else:
            # record & render user
            st.session_state.history.append(("user", user_input))
            st.chat_message("user").write(user_input)

            # build history_pairs [(q,a), ...]
            history_pairs = []
            flat = st.session_state.history
            for i in range(0, len(flat) - 1, 2):
                if flat[i][0] == "user" and flat[i+1][0] == "assistant":
                    history_pairs.append((flat[i][1], flat[i+1][1]))

            # run the conversational chain
            res = st.session_state.qa({
                "question": user_input,
                "chat_history": history_pairs
            })
            answer = res["answer"]

            # record & render assistant
            st.session_state.history.append(("assistant", answer))
            st.chat_message("assistant").write(answer)

            # show sources
            with st.expander("Sources", expanded=False):
                for doc in res["source_documents"]:
                    src = doc.metadata.get("source", "unknown")
                    snippet = doc.page_content.strip().replace("\n", " ")[:300]
                    st.markdown(f"- **{src}**: {snippet}…")
