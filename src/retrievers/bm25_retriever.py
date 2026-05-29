from rank_bm25 import BM25Okapi


class BM25Retriever:

    def __init__(self, documents):

        self.documents = documents

        self.texts = [

            doc.page_content for doc in documents
        ]

        self.tokenized_texts = [

            text.lower().split()

            for text in self.texts
        ]

        self.bm25 = BM25Okapi(

            self.tokenized_texts
        )

    def retrieve(

        self,

        query,

        top_k=4
    ):

        tokenized_query = (

            query.lower().split()
        )

        scores = self.bm25.get_scores(

            tokenized_query
        )

        ranked_indices = sorted(

            range(len(scores)),

            key=lambda i: scores[i],

            reverse=True
        )[:top_k]

        return [

            self.documents[i]

            for i in ranked_indices
        ]