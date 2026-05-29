def build_chat_history(

    chat_history
):

    history_text = ""

    for item in chat_history:

        history_text += (

            f"User: {item['question']}\n"
        )

        history_text += (

            f"Assistant: {item['answer']}\n"
        )

    return history_text