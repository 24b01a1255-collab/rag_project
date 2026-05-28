import os
import shutil
import tempfile

import streamlit as st

from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from langchain_core.documents import Document
# ---------------------------------------------------
# OCR IMPORTS
# ---------------------------------------------------

from pdf2image import convert_from_path

import pytesseract

# ---------------------------------------------------
# CHROMA
# ---------------------------------------------------

from src.vectorstore.chroma_store import (

    create_chroma_vectorstore,

    load_chroma_vectorstore,

    get_chroma_retriever
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(

    page_title="OCR RAG System",

    layout="wide"
)

st.title("OCR Enabled RAG System")

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "retriever" not in st.session_state:

    st.session_state.retriever = None

if "messages" not in st.session_state:

    st.session_state.messages = []

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

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
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("Database")

# ---------------------------------------------------
# CLEAR DATABASE
# ---------------------------------------------------

if st.sidebar.button(

    "Clear Chroma Database"
):

    if os.path.exists("chroma_db"):

        shutil.rmtree("chroma_db")

    st.session_state.retriever = None

    st.success("ChromaDB Cleared")

# ---------------------------------------------------
# LOAD EXISTING DB
# ---------------------------------------------------

if (

    os.path.exists("chroma_db")

    and st.session_state.retriever is None
):

    try:

        vectorstore = load_chroma_vectorstore(

            embeddings
        )

        retriever = get_chroma_retriever(

            vectorstore
        )

        st.session_state.retriever = retriever

        st.sidebar.success(

            "Persistent DB Loaded"
        )

    except Exception as e:

        st.sidebar.error(

            f"Error: {e}"
        )

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_files = st.file_uploader(

    "Upload PDFs",

    type=["pdf"],

    accept_multiple_files=True
)

# ---------------------------------------------------
# OCR FUNCTION
# ---------------------------------------------------

def extract_text_using_ocr(pdf_path, file_name):

    documents = []

    images = convert_from_path(pdf_path)

    for i, image in enumerate(images):

        text = pytesseract.image_to_string(image)

        if text.strip():

            doc = Document(

                page_content=text,

                metadata={

                    "source": file_name,

                    "page": i + 1
                }
            )

            documents.append(doc)

    return documents

# ---------------------------------------------------
# PROCESS DOCUMENTS
# ---------------------------------------------------

if uploaded_files:

    all_documents = []

    for uploaded_file in uploaded_files:

        with tempfile.NamedTemporaryFile(

            delete=False,

            suffix=".pdf"

        ) as tmp_file:

            tmp_file.write(

                uploaded_file.read()
            )

            temp_path = tmp_file.name

        # ---------------------------------------------------
        # TRY NORMAL PDF EXTRACTION
        # ---------------------------------------------------

        loader = PyPDFLoader(

            temp_path
        )

        documents = loader.load()

        extracted_text = ""

        for doc in documents:

            extracted_text += doc.page_content.strip()

        # ---------------------------------------------------
        # OCR FALLBACK
        # ---------------------------------------------------

        if len(extracted_text) < 50:

            st.warning(

                f"OCR Used For {uploaded_file.name}"
            )

            documents = extract_text_using_ocr(

                temp_path,

                uploaded_file.name
            )

        else:

            for doc in documents:

                doc.metadata["source"] = (

                    uploaded_file.name
                )

        all_documents.extend(

            documents
        )

    # ---------------------------------------------------
    # SPLITTER
    # ---------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200
    )

    split_docs = splitter.split_documents(

        all_documents
    )

    # ---------------------------------------------------
    # CREATE CHROMA
    # ---------------------------------------------------

    vectorstore = create_chroma_vectorstore(

        split_docs,

        embeddings
    )

    retriever = get_chroma_retriever(

        vectorstore
    )

    st.session_state.retriever = retriever

    st.success(

        "Documents Stored Successfully"
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

            "Upload PDFs First"
        )

    else:

        with st.chat_message("assistant"):

            docs = st.session_state.retriever.invoke(

                question
            )

            # ---------------------------------------------------
            # FILTER LOW RELEVANCE DOCS
            # ---------------------------------------------------

            filtered_docs = []

            for doc in docs:

                if len(doc.page_content.strip()) > 50:

                    filtered_docs.append(doc)

            context = ""

            for doc in filtered_docs:

                context += (

                    doc.page_content + "\n"
                )

            history_text = ""

            for item in st.session_state.chat_history:

                history_text += (

                    f"User: {item['question']}\n"
                )

                history_text += (

                    f"Assistant: {item['answer']}\n"
                )

            prompt = f"""

You are a RAG assistant.

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
"""

            response = llm.invoke(

                prompt
            )

            answer = response.content

            st.session_state.chat_history.append({

                "question": question,

                "answer": answer
            })

            st.markdown(answer)

            st.session_state.messages.append({

                "role": "assistant",

                "content": answer
            })

            # ---------------------------------------------------
            # SHOW SOURCES
            # ---------------------------------------------------

            if (

                "could not find" not in answer.lower()

                and len(filtered_docs) > 0
            ):

                st.markdown("### Sources")

                shown_sources = set()

                for doc in filtered_docs:

                    source = doc.metadata.get(

                        "source",

                        "Unknown"
                    )

                    page = doc.metadata.get(

                        "page",

                        "N/A"
                    )

                    source_text = (

                        f"{source} — Page {page}"
                    )

                    if source_text not in shown_sources:

                        st.markdown(

                            f"- {source_text}"
                        )

                        shown_sources.add(

                            source_text
                        )