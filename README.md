# RAG Chatbot using Mistral (Ollama)

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents and ask questions based on their content.

Instead of relying only on a language model’s internal knowledge, the system retrieves relevant information from the uploaded document and uses it to generate accurate, context-aware answers.

---

## Features

* Upload and process PDF documents
* Semantic search using embeddings
* Context-aware answer generation using Mistral via Ollama
* Interactive chat interface built with Streamlit
* Local vector database using ChromaDB
* Fully local setup with no dependency on external APIs

---

## How it Works

### 1. Ingestion Pipeline

* Loads the PDF document
* Splits it into smaller chunks
* Converts chunks into embeddings
* Stores them in a vector database

### 2. Retrieval

* Converts user query into an embedding
* Finds the most relevant chunks using similarity search

### 3. Generation

* Combines retrieved context with the query
* Generates a response using the Mistral model

---

## Tech Stack

* Python
* LangChain
* Ollama (Mistral)
* ChromaDB
* HuggingFace Embeddings
* Streamlit

---

## Project Structure

```
RAG/
│── app.py                # Streamlit frontend
│── ingestion.py          # PDF ingestion pipeline
│── retrieval.py          # Query and response logic
│── requirements.txt
│── .gitignore
```

---

## Setup Instructions

### 1. Clone the repository

```
git clone https://github.com/your-username/your-repo.git
cd RAG
```

### 2. Create a virtual environment

```
python -m venv rag_env
rag_env\Scripts\activate   # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

### 4. Install and run Ollama

Install Ollama and pull the model:

```
ollama pull mistral
```

### 5. Run the application

```
streamlit run app.py
```

---

## Usage

1. Upload a PDF using the sidebar
2. Enter a question related to the document
3. The system retrieves relevant content and generates an answer

---

## Notes

* The vector database (`chroma_db`) is generated locally and is not stored in the repository
* Large files and environments are excluded using `.gitignore`
* Ensure Ollama is running before starting the application

---

## Future Improvements

* Support for multiple documents
* Source attribution (page-level citations)
* Improved retrieval ranking
* Deployment to a public interface

---

## Author

Chetan Misquith
