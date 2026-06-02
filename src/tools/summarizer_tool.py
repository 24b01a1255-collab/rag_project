from langchain_groq import ChatGroq


def summarize_documents(documents, llm):

    if not documents:
        return "No documents available."

    text = ""

    for doc in documents[:20]:
        text += doc.page_content + "\n"

    prompt = f"""
You are a document summarization assistant.

Generate:

1. Executive Summary
2. Key Points
3. Important Insights

Document Content:

{text}
"""

    response = llm.invoke(prompt)

    return response.content