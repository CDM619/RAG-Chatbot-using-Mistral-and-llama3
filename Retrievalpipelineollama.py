from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load LLM
llm = OllamaLLM(model="mistral")

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Load vector DB
db = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

def ask(query):
    docs = db.similarity_search(query, k=3)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
Answer only from the context.

Context:
{context}

Question:
{query}
"""

    return llm.invoke(prompt)

while True:
    q = input("Ask: ")

    if q == "exit":
        break

    print(ask(q))