from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os


# ---------- CONFIG ----------
PDF_PATH = "Google.pdf"
DB_DIR = "chroma_db"

# ---------- 1. Load LLM ----------
llm = OllamaLLM(model="mistral")

# ---------- 2. Embeddings ----------
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# ---------- 3. Load or Create DB ----------
if os.path.exists(DB_DIR):
    print("✅ Loading existing vector DB...")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

else:
    print("📄 Creating vector DB (first run)...")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=DB_DIR
    )
    db.persist()

    print("✅ DB created and saved.")

# ---------- 4. RAG function ----------
def ask(query):
    docs = db.max_marginal_relevance_search(query, k=4, fetch_k=10)

    context_parts = []
    sources = []

    for doc in docs:
        context_parts.append(doc.page_content)
        if "page" in doc.metadata:
            sources.append(f"Page {doc.metadata['page']}")

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful and conversational AI assistant.

Rules:
- Answer ONLY using the context
- If not found, say: "I don't know based on the document"
- Keep answers clear and natural

Context:
{context}

User: {query}
Assistant:
"""

    answer = llm.invoke(prompt)

    return f"{answer}\n\nSources: {', '.join(set(sources))}"

# ---------- 5. Chat loop ----------
while True:
    q = input("\nAsk something (or type 'exit'): ")
    if q.lower() == "exit":
        break

    print("\n", ask(q))