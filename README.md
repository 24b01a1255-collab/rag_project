## Document Upload Workflow:
           Upload Document
           
           ↓
           
           Loader
           
           ↓
           
           Text Splitter
           
           ↓
           
           Chunk Creation
           
           ↓
           
           Embedding Model
           
           ↓
           
           Vector Embeddings
           
           ↓
           
           Store in ChromaDB     

## Retrieval Workflow:
           Question
           
           ↓
           
           Expand Query
           
           ↓
           
           Chroma Search
           
           ↓
           
           BM25 Search
           
           ↓
           
           Merge
           
           ↓
           
           Reranker
           
           ↓
           
           Top Chunks
           
           ↓
           
           LLM
           
           ↓
           
           Answer

# Advanced Hybrid RAG System

An advanced Retrieval-Augmented Generation (RAG) application built using **LangChain**, **ChromaDB**, **BM25**, **Groq LLM**, and **Streamlit**. The system combines dense vector retrieval with sparse keyword search, query expansion, reranking, OCR support, document summarization, and conversational memory to provide accurate and context-aware responses from multiple document formats.


## Features

1. Multiple Document Support
2. Automatic Text Spitting
3. Embedding Generation
4. Chroma DB
5. Hybrid Retrieval
6. Query Expansion
7. Cross Encoder Reranker
8. Prompt Engineering
9. Conversation Memory
10. Voice Input
11. Document Summarization
12. Web Search Fallaback
13. Source Display


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


