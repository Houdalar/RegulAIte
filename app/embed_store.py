from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def build_or_load_store(chunks=None, persist_dir="vectorstore"):
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if chunks is not None:
        db = Chroma.from_documents(chunks, embedding=embed, persist_directory=persist_dir)
    else:
        db = Chroma(persist_directory=persist_dir, embedding_function=embed)
    return db
