from langchain_community.vectorstores import Chroma

# ---------------------------------------------------
# CHROMA DIRECTORY
# ---------------------------------------------------

CHROMA_PATH = "chroma_db"

# ---------------------------------------------------
# CREATE CHROMA DB
# ---------------------------------------------------

def create_chroma_vectorstore(

    documents,

    embeddings
):

    vectorstore = Chroma.from_documents(

        documents=documents,

        embedding=embeddings,

        persist_directory=CHROMA_PATH
    )

    vectorstore.persist()

    return vectorstore

# ---------------------------------------------------
# LOAD CHROMA DB
# ---------------------------------------------------

def load_chroma_vectorstore(

    embeddings
):

    vectorstore = Chroma(

        persist_directory=CHROMA_PATH,

        embedding_function=embeddings
    )

    return vectorstore

# ---------------------------------------------------
# RETRIEVER
# ---------------------------------------------------

def get_chroma_retriever(

    vectorstore
):

    retriever = vectorstore.as_retriever(

        search_kwargs={"k": 3}
    )

    return retriever