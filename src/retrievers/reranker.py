from sentence_transformers import (
    CrossEncoder
)


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(

            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(

        self,

        query,

        docs,

        top_k=4
    ):

        pairs = [

            (query, doc.page_content)

            for doc in docs
        ]

        scores = self.model.predict(

            pairs
        )

        scored_docs = list(

            zip(docs, scores)
        )

        scored_docs.sort(

            key=lambda x: x[1],

            reverse=True
        )

        return [

            doc for doc, score

            in scored_docs[:top_k]
        ]