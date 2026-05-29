from src.retrievers.bm25_retriever import (
    BM25Retriever
)


class HybridRetriever:

    def __init__(

        self,

        vectorstore,

        documents
    ):

        self.vectorstore = vectorstore

        self.bm25 = BM25Retriever(

            documents
        )

    def retrieve(

        self,

        query,

        top_k=4
    ):

        vector_docs = self.vectorstore.similarity_search(

            query,

            k=top_k
        )

        bm25_docs = self.bm25.retrieve(

            query,

            top_k
        )

        combined = []

        seen = set()

        for doc in vector_docs + bm25_docs:

            text = doc.page_content

            if text not in seen:

                combined.append(doc)

                seen.add(text)

        return combined[:top_k]