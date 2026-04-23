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

    prompt = prompt = f"""
You are a strict retrieval-based AI assistant.

You must follow these rules exactly:

1. Answer ONLY using the information explicitly provided in the context.
2. Do NOT use prior knowledge, assumptions, or external information.
3. Do NOT infer or guess missing details.
4. If the answer is not explicitly present in the context, respond ONLY with:
   "I don't know"
5. Do NOT add explanations beyond what is stated in the context.
6. If the context is incomplete or ambiguous, say:
   "I don't know"

Context:
{context}

Question:
{query}

Answer:
"""

    return llm.invoke(prompt)

while True:
    q = input("Ask: ")

    if q == "exit":
        break

    print(ask(q))