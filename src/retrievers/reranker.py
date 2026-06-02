from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    def rerank(
        self,
        query,
        docs,
        top_k=4,
        threshold=0.30
    ):

        if not docs:
            return []

        pairs = [
            (query, doc.page_content)
            for doc in docs
        ]

        scores = self.model.predict(pairs)

        scored_docs = list(
            zip(docs, scores)
        )

        scored_docs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        filtered_docs = [
            doc
            for doc, score in scored_docs
            if score >= threshold
        ]

        return filtered_docs[:top_k]