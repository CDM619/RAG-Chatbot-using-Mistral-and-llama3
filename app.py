import streamlit as st
import tempfile
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="Document Intelligence RAG System", layout="wide")

st.title(" Document Intelligence RAG System ")

# -------- Sidebar Upload --------
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

# -------- Initialize DB --------
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if "db" not in st.session_state:
    st.session_state.db = None

# -------- Process PDF --------
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.sidebar.success("Processing PDF...")

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    db = Chroma.from_documents(chunks, embeddings)

    st.session_state.db = db
    st.sidebar.success("PDF loaded successfully!")

# -------- LLM --------
llm = OllamaLLM(model="mistral")

# -------- Chat UI --------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

query = st.chat_input("Ask something about your document...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)

    if st.session_state.db is None:
        response = "Please upload a PDF first."
    else:
        docs = st.session_state.db.similarity_search(query, k=3)

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
Answer ONLY from the context.

Context:
{context}

Question:
{query}
"""

        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
