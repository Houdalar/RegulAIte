from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_and_split(data_dir: str, chunk_size: int = 500, chunk_overlap: int = 50):
    docs = []
    for path in Path(data_dir).glob("*"):
        if path.suffix.lower() == ".pdf":
            docs += PyPDFLoader(str(path)).load()
        else:
            docs += TextLoader(str(path)).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)
