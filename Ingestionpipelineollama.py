from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

# -------- CONFIG --------
PDF_PATH = "Google.pdf"
DB_DIR = "chroma_db"

# -------- 1. Load PDF --------
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Loaded {len(documents)} pages")

# -------- 2. Split --------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# -------- 3. Embeddings --------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# -------- 4. Store in Vector DB --------
db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=DB_DIR
)

db.persist()

print("✅ Ingestion complete. Vector DB saved.")