
import streamlit as st

from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_community.vectorstores import (
    FAISS
)

from langchain_community.embeddings import (
    HuggingFaceEmbeddings
)

import tempfile

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(

    page_title="Basic RAG System",

    layout="wide"
)

st.title("Basic RAG System")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(

    "Upload PDF",

    type=["pdf"]
)

# ---------------------------------------------------
# LOAD LLM
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
# PROCESS PDF
# ---------------------------------------------------

if uploaded_file:

    with tempfile.NamedTemporaryFile(

        delete=False,

        suffix=".pdf"

    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        temp_path = tmp_file.name

    # LOAD PDF

    loader = PyPDFLoader(temp_path)

    documents = loader.load()

    # ---------------------------------------------------
    # SPLIT DOCUMENTS
    # ---------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=200
    )

    split_docs = splitter.split_documents(

        documents
    )

    # ---------------------------------------------------
    # CREATE VECTOR STORE
    # ---------------------------------------------------

    vectorstore = FAISS.from_documents(

        split_docs,

        embeddings
    )

    retriever = vectorstore.as_retriever()

    st.success("PDF Processed Successfully")

    # ---------------------------------------------------
    # QUESTION INPUT
    # ---------------------------------------------------

    question = st.chat_input(

        "Ask Question"
    )

    # ---------------------------------------------------
    # QUESTION ANSWERING
    # ---------------------------------------------------

    if question:

        st.write("User Question:")

        st.write(question)

        # RETRIEVE DOCUMENTS

        docs = retriever.invoke(question)

        # BUILD CONTEXT

        context = ""

        for doc in docs:

            context += doc.page_content + "\n"

        # CREATE PROMPT

        prompt = f"""
Answer the question only from the provided context.

Context:
{context}

Question:
{question}
"""

        # GENERATE RESPONSE

        response = llm.invoke(prompt)

        # SHOW ANSWER

        st.write("Answer:")

        st.write(response.content)

