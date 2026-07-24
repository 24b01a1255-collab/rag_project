Work Flow Diagram:
           
           Document Upload
                  │
                  ▼
        Document Loader
                  │
      ┌───────────┴───────────┐
      │                       │
      ▼                       ▼
 OCR (if needed)        Text Extraction
      │                       │
      └───────────┬───────────┘
                  ▼
        Generate Embeddings
                  │
      ┌───────────┴───────────┐
      ▼                       ▼
 ChromaDB Index         BM25 Index

──────────────────────────────────────────────

              User Question
                    │
                    ▼
          Conversation Memory
                    │
                    ▼
           Query Expansion
                    │
                    ▼
      Hybrid Retrieval (BM25 + Chroma)
                    │
                    ▼
      Cross-Encoder Reranker
                    │
         Low Confidence?
          │              │
         No             Yes
          │              ▼
          │        Web Search Tool
          │              │
          └──────┬───────┘
                 ▼
             Groq LLM
                 │
                 ▼
      Answer + Source Citations
                 │
                 ▼
        Update Conversation Memory

# Advanced Hybrid RAG System

An advanced Retrieval-Augmented Generation (RAG) application built using **LangChain**, **ChromaDB**, **BM25**, **Groq LLM**, and **Streamlit**. The system combines dense vector retrieval with sparse keyword search, query expansion, reranking, OCR support, document summarization, and conversational memory to provide accurate and context-aware responses from multiple document formats.


## Features

- Hybrid Retrieval (Vector Search + BM25)
- Semantic Search using ChromaDB
- Query Expansion for improved retrieval
- Cross-Encoder Reranking
- Persistent Vector Database
- Multi-file Document Support
- Conversational Memory
- Source Citation
- OCR Support using Tesseract
- PDF Image Extraction with Poppler
- Document Summarization Tool
- Web Search Tool
- Voice Search Support
- Modular Project Architecture
- Streamlit User Interface


## Supported File Formats

- PDF
- DOCX
- PPTX
- TXT
- CSV
- Images (OCR)



## Project Architecture

```
rag_pro/
│
├── app.py
├── requirements.txt
├── chroma_db/
├── src/
│   ├── config/
│   ├── database/
│   ├── loaders/
│   ├── memory/
│   ├── prompts/
│   ├── retrievers/
│   ├── tools/
│   ├── utils/
│   └── vectorstore/
│
└── .streamlit/
    └── secrets.toml
```

---

## Retrieval Pipeline

```
User Query
      │
      ▼
Query Expansion
      │
      ▼
Hybrid Retrieval
(BM25 + ChromaDB)
      │
      ▼
Cross Encoder Reranker
      │
      ▼
Relevant Context
      │
      ▼
Groq LLM
      │
      ▼
Final Answer
      │
      ▼
Sources Returned
```



## Project Modules

### Document Loader

- Loads multiple document formats
- OCR support for scanned PDFs and images

### Vector Store

- Persistent ChromaDB storage
- Embedding generation

### Hybrid Retriever

- Dense semantic retrieval
- BM25 keyword retrieval
- Result fusion

### Reranker

- Cross-Encoder reranking
- Improves retrieval relevance

### Memory

- Maintains conversational context

### Document Summarizer

- Generates concise summaries of uploaded documents

### Web Search

- Retrieves external information when required

### Voice Search

- Allows speech-based user queries


## OCR Support

Scanned PDFs and images are processed using:

- Tesseract OCR
- Poppler
- Pillow

This enables text extraction from image-based documents.

