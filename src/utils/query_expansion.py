def expand_query(query):

    expansions = {

        "ml": "machine learning ai models",

        "ai": "artificial intelligence machine learning",

        "dbms": "database management system sql",

        "os": "operating system scheduling memory",

        "cn": "computer networks tcp ip"
    }

    lower_query = query.lower()

    for key in expansions:

        if key in lower_query:

            query += " " + expansions[key]

    return query