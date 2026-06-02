import mysql.connector


def get_connection():

    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="rag_db"
    )


def save_chat(question, answer):

    conn = get_connection()

    cursor = conn.cursor()

    query = """
    INSERT INTO chat_history
    (question, answer)
    VALUES (%s, %s)
    """

    cursor.execute(
        query,
        (question, answer)
    )

    conn.commit()

    cursor.close()
    conn.close()


def get_all_chats():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM chat_history"
    )

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return rows