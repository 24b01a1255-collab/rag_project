import os

import shutil

import streamlit as st

from langchain_groq import ChatGroq

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

# ---------------------------------------------------
# LOADERS
# ---------------------------------------------------

from src.loaders.loader_router import (
    load_document
)

# ---------------------------------------------------
# CHROMA
# ---------------------------------------------------

from src.vectorstore.chroma_store import (

    create_chroma_vectorstore
)

# ---------------------------------------------------
# HYBRID RETRIEVER
# ---------------------------------------------------

from src.retrievers.hybrid_retriever import (
    HybridRetriever
)

from src.retrievers.reranker import (
    Reranker
)
from src.tools.summarizer_tool import (
    summarize_documents
)
# ---------------------------------------------------
# UTILS
# ---------------------------------------------------

from src.utils.query_expansion import (
    expand_query
)

from src.utils.source_formatter import (
    format_sources
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(

    page_title="Advanced Hybrid RAG",

    layout="wide"
)

st.title("Advanced Hybrid RAG System")

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "retriever" not in st.session_state:

    st.session_state.retriever = None

if "messages" not in st.session_state:

    st.session_state.messages = []

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

if "documents" not in st.session_state:

    st.session_state.documents = []

if "summary" not in st.session_state:

    st.session_state.summary = ""

# ---------------------------------------------------
# LLM
# ---------------------------------------------------

llm = ChatGroq(

    groq_api_key=st.secrets["GROQ_API_KEY"],

    model_name="llama-3.1-8b-instant"
)

# ---------------------------------------------------
# EMBEDDINGS
# ---------------------------------------------------

embeddings = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------
# RERANKER
# ---------------------------------------------------

reranker = Reranker()

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("Database")

# ---------------------------------------------------
# DOCUMENT SUMMARY TOOL
# ---------------------------------------------------

if st.sidebar.button("Generate Document Summary"):

    if not st.session_state.documents:

        st.warning("Upload documents first")

    else:

        with st.spinner("Generating summary..."):

            summary = summarize_documents(

                st.session_state.documents,

                llm
            )

            st.session_state.summary = summary

            st.session_state.messages.append({

                "role": "assistant",

                "content": summary
            })

            st.success("Summary Generated")

if st.sidebar.button(

    "Clear Chroma Database"
):

    if os.path.exists("chroma_db"):

        shutil.rmtree("chroma_db")

        st.session_state.retriever = None

        st.session_state.documents = []

        st.session_state.summary = ""

        st.success("Database Cleared")

# ---------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------

uploaded_files = st.file_uploader(

    "Upload Documents",

    type=[

        "pdf",

        "docx",

        "csv",

        "pptx"
    ],

    accept_multiple_files=True
)

# ---------------------------------------------------
# PROCESS DOCUMENTS
# ---------------------------------------------------

if uploaded_files:

    all_documents = []

    for uploaded_file in uploaded_files:

        documents = load_document(

            uploaded_file
        )

        all_documents.extend(

            documents
        )

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200
    )

    split_docs = splitter.split_documents(

        all_documents
    )

    vectorstore = create_chroma_vectorstore(

        split_docs,

        embeddings
    )

    hybrid_retriever = HybridRetriever(

        vectorstore,

        split_docs
    )

    st.session_state.retriever = (

        hybrid_retriever
    )

    st.session_state.documents = (

        split_docs
    )

    st.success(

        "Documents Stored Successfully"
    )
# ---------------------------------------------------
# SHOW DOCUMENT SUMMARY
# ---------------------------------------------------

if st.session_state.summary:

    st.subheader("Document Summary")

    st.markdown(

        st.session_state.summary
    )

# ---------------------------------------------------
# DISPLAY CHAT
# ---------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(

        message["role"]
    ):

        st.markdown(

            message["content"]
        )

# ---------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------

question = st.chat_input(

    "Ask Question"
)

# ---------------------------------------------------
# QUESTION ANSWERING
# ---------------------------------------------------

if question:

    st.session_state.messages.append({

        "role": "user",

        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    if st.session_state.retriever is None:

        st.warning(

            "Upload Documents First"
        )

    else:

        with st.chat_message("assistant"):

            # ---------------------------------------------------
            # QUERY EXPANSION
            # ---------------------------------------------------

            expanded_query = expand_query(

                question
            )

            # ---------------------------------------------------
            # HYBRID RETRIEVAL
            # ---------------------------------------------------

            docs = st.session_state.retriever.retrieve(

                expanded_query,

                top_k=6
            )

            # ---------------------------------------------------
            # RERANKING
            # ---------------------------------------------------

            docs = reranker.rerank(

                question,

                docs,

                top_k=4
            )

            # ---------------------------------------------------
            # CONTEXT
            # ---------------------------------------------------

            context = ""

            for doc in docs:

                context += (

                    doc.page_content + "\n"
                )

            # ---------------------------------------------------
            # MEMORY
            # ---------------------------------------------------

            history_text = ""

            for item in st.session_state.chat_history[-5:]:

                history_text += (

                    f"User: {item['question']}\n"
                )

                history_text += (

                    f"Assistant: {item['answer']}\n"
                )

            # ---------------------------------------------------
            # PROMPT
            # ---------------------------------------------------

            prompt = f"""

You are an advanced RAG assistant.

Answer ONLY from the context.

Conversation History:
{history_text}

Context:
{context}

Question:
{question}

Rules:

1. If answer is unavailable say:
"I could not find relevant information."

2. Do not hallucinate.

3. Give concise answers.

4. Use context strictly.
"""

            # ---------------------------------------------------
            # GENERATE RESPONSE
            # ---------------------------------------------------

            response = llm.invoke(

                prompt
            )

            answer = response.content

            # ---------------------------------------------------
            # SAVE MEMORY
            # ---------------------------------------------------

            st.session_state.chat_history.append({

                "question": question,

                "answer": answer
            })

            # ---------------------------------------------------
            # SHOW ANSWER
            # ---------------------------------------------------

            st.markdown(answer)

            st.session_state.messages.append({

                "role": "assistant",

                "content": answer
            })

            # ---------------------------------------------------
            # SHOW SOURCES
            # ---------------------------------------------------

            if (

                "could not find"

                not in answer.lower()
            ):

                st.markdown("### Sources")

                sources = format_sources(docs)

                for source in sources:

                    st.markdown(

                        f"- {source}"
                    )