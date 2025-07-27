from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate

from .llm_factory import get_llm


def create_qa_chain(persist_dir: str, provider: str = "openai"):
    # 1️⃣ Embeddings (must match ingest)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2️⃣ Vector store
    db = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # 3️⃣ LLM
    llm = get_llm(provider)

    # 4️⃣ Question condenser prompt & chain
    condense_prompt = PromptTemplate(
        template="""
Given the following conversation and a follow-up question, rephrase the follow-up to be a standalone query.

Conversation History:
{chat_history}

Follow Up Question:
{question}

Standalone question:""",
        input_variables=["chat_history", "question"],
    )
    condense_chain = LLMChain(llm=llm, prompt=condense_prompt)

    # 5️⃣ Combine-documents prompt (for final answer)
    combine_prompt = PromptTemplate(
        template="""
    You are **RegulAIte**, a friendly policy assistant.
    Your job is to help users understand the policies in the uploaded documents.

    When the user’s question relates to “policies” in general—e.g. “What are the workplace policies?”—you should automatically:

    1. **Detect** all subsection titles in the excerpts that pertain to policies (e.g., “Workplace Harassment,” “Workplace Violence,” etc.).
    2. **Summarize** each one in a single sentence.
    3. **Present** your answer as a **numbered list** with each title in bold, followed by its summary.
    4. **Do not** include any extraneous greetings or apologies—just the list.

    For any other question:

    - If it’s covered explicitly, answer directly and naturally.
    - If it isn’t covered word‑for‑word, start with:
    > “While this isn’t stated word‑for‑word, based on the excerpts:”
    then give your best answer.

    If the user says something unrelated to the file (e.g. “hello”), respond:
    > “I’m RegulAIte, your policy assistant. Please ask me about the policies in your uploaded documents.”

    **Excerpts:**
    {context}

    **Question:**
    {question}

    **Answer:**""",
        input_variables=["context", "question"],
    )

    # No LLMChain here—will pass prompt via combine_docs_chain_kwargs

    # 6️⃣ Build the conversational chain with correct parameters
    return ConversationalRetrievalChain.from_llm(
        llm,  # Base LLM
        db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
        ),  # Retriever
        condense_question_prompt=condense_prompt,  # Condense prompt
        combine_docs_chain_kwargs={"prompt": combine_prompt},  # Final-answer prompt
        return_source_documents=True,
    )
