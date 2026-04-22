from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# -------- CONFIG --------
DB_DIR = "chroma_db"

def ingest_pdf(file_path):
    print(f"📄 Loading: {file_path}")

    # 1. Load PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    # 4. Store / Update DB
    if os.path.exists(DB_DIR):
        db = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )
        db.add_documents(chunks)
    else:
        db = Chroma.from_documents(
            chunks,
            embeddings,
            persist_directory=DB_DIR
        )

    db.persist()
    print("✅ Ingestion complete!")

# -------- CLI Usage --------
if __name__ == "__main__":
    file_path = input("Enter PDF path: ").strip()
    ingest_pdf(file_path)