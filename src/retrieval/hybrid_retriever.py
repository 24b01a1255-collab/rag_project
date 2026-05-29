from rank_bm25 import BM25Okapi

class HybridRetriever:

    def __init__(

        self,

        documents,

        vector_retriever
    ):

        self.documents = documents

        self.vector_retriever = vector_retriever

        self.tokenized_docs = [

            doc.page_content.split()

            for doc in documents
        ]

        self.bm25 = BM25Okapi(

            self.tokenized_docs
        )

    def retrieve(

        self,

        query,

        k=4
    ):

        # -----------------------------
        # VECTOR SEARCH
        # -----------------------------

        vector_docs = self.vector_retriever.invoke(

            query
        )

        # -----------------------------
        # BM25 SEARCH
        # -----------------------------

        tokenized_query = query.split()

        bm25_scores = self.bm25.get_scores(

            tokenized_query
        )

        bm25_top_indices = sorted(

            range(len(bm25_scores)),

            key=lambda i: bm25_scores[i],

            reverse=True
        )[:k]

        bm25_docs = [

            self.documents[i]

            for i in bm25_top_indices
        ]

        # -----------------------------
        # COMBINE RESULTS
        # -----------------------------

        combined_docs = []

        seen = set()

        for doc in vector_docs + bm25_docs:

            text = doc.page_content

            if text not in seen:

                combined_docs.append(doc)

                seen.add(text)

        return combined_docs[:k]