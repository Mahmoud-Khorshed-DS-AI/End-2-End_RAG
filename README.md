# End-to-End PDF RAG with Qwen + FAISS + MLOps

This project is a teaching-ready production-style RAG system for unstructured PDF documents.

## What it does

- Loads PDF documents
- Falls back to OCR for scanned PDFs
- Adds metadata: source file, page number, page description
- Creates semantic-like chunks
- Embeds chunks using `sentence-transformers/all-MiniLM-L6-v2`
- Stores vectors in FAISS
- Uses Qwen open-source LLM for grounded answers
- Shows sources and page numbers in Streamlit
- Includes evaluation and CI/CD tests

## Architecture

```text
PDF → OCR/Text Extraction → Metadata → Chunking → Embeddings → FAISS
→ Retriever → Qwen LLM → Answer + Sources → Streamlit UI
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For OCR support on Ubuntu:

```bash
sudo apt-get install -y tesseract-ocr
```

## Run Streamlit

```bash
streamlit run app/streamlit_app.py
```

## Run tests

```bash
pytest tests/
```

## Docker

```bash
docker build -t qwen-pdf-rag .
docker run -p 8501:8501 qwen-pdf-rag
```

## Evaluation

Create golden questions in:

```text
evaluation/golden_questions.csv
```

Measure whether the correct page appears in top-k retrieved chunks.

## Teaching Flow

1. Basic RAG
2. PDF ingestion + OCR
3. Metadata and page reference
4. Semantic chunking
5. FAISS retrieval
6. Qwen generation
7. Streamlit app
8. Evaluation
9. CI/CD
10. GitHub deployment lifecycle
