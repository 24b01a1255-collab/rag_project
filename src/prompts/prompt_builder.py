def build_prompt(

    history_text,

    context,

    question
):

    prompt = f"""

You are a Hybrid OCR RAG assistant.

Answer ONLY from the given context.

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

4. Prefer exact text when available.
"""

    return prompt