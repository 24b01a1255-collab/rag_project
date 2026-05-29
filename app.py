import os
import shutil
import tempfile

import streamlit as st

from langchain_groq import ChatGroq

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

from src.vectorstore.chroma_store import (

    create_chroma_vectorstore,

    load_chroma_vectorstore,

    get_chroma_retriever
)

from src.loaders.ocr_loader import (
    extract_documents
)

from src.retrieval.hybrid_retriever import (
    HybridRetriever
)

from src.prompts.prompt_builder import (
    build_prompt
)

from src.memory.memory_manager import (
    build_chat_history
)

from src.utils.source_formatter import (
    show_sources
)

# ---------------------------------------------------

st.set_page_config(

    page_title="Refactored RAG System",

    layout="wide"
)

st.title("Refactored Hybrid OCR RAG")

# ---------------------------------------------------

if "retriever" not in st.session_state:

    st.session_state.retriever = None

if "hybrid_retriever" not in st.session_state:

    st.session_state.hybrid_retriever = None

if "messages" not in st.session_state:

    st.session_state.messages = []

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

# ---------------------------------------------------

llm = ChatGroq(

    groq_api_key=st.secrets["GROQ_API_KEY"],

    model_name="llama-3.1-8b-instant"
)

embeddings = HuggingFaceEmbeddings(

    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------

st.sidebar.title("Database")

if st.sidebar.button(

    "Clear Chroma Database"
):

    if os.path.exists("chroma_db"):

        shutil.rmtree("chroma_db")

    st.session_state.retriever = None

    st.session_state.hybrid_retriever = None

    st.success("Database Cleared")

# ---------------------------------------------------

uploaded_files = st.file_uploader(

    "Upload PDFs",

    type=["pdf"],

    accept_multiple_files=True
)

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

        documents = extract_documents(

            temp_path,

            uploaded_file.name
        )

        all_documents.extend(documents)

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

    retriever = get_chroma_retriever(

        vectorstore
    )

    hybrid_retriever = HybridRetriever(

        split_docs,

        retriever
    )

    st.session_state.retriever = retriever

    st.session_state.hybrid_retriever = hybrid_retriever

    st.success(

        "Documents Stored Successfully"
    )

# ---------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(

        message["role"]
    ):

        st.markdown(

            message["content"]
        )

# ---------------------------------------------------

question = st.chat_input(

    "Ask Question"
)

# ---------------------------------------------------

if question:

    st.session_state.messages.append({

        "role": "user",

        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    if st.session_state.hybrid_retriever is None:

        st.warning(

            "Upload PDFs First"
        )

    else:

        with st.chat_message("assistant"):

            docs = st.session_state.hybrid_retriever.retrieve(

                question
            )

            filtered_docs = []

            for doc in docs:

                if len(doc.page_content.strip()) > 50:

                    filtered_docs.append(doc)

            context = ""

            for doc in filtered_docs:

                context += (

                    doc.page_content + "\n"
                )

            history_text = build_chat_history(

                st.session_state.chat_history
            )

            prompt = build_prompt(

                history_text,

                context,

                question
            )

            response = llm.invoke(

                prompt
            )

            answer = response.content

            st.markdown(answer)

            st.session_state.messages.append({

                "role": "assistant",

                "content": answer
            })

            st.session_state.chat_history.append({

                "question": question,

                "answer": answer
            })

            if (

                "could not find"

                not in answer.lower()

                and len(filtered_docs) > 0
            ):

                show_sources(

                    filtered_docs,

                    st
                )