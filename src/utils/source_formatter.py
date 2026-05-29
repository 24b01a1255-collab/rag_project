def format_sources(docs):

    formatted_sources = []

    shown_sources = set()

    for doc in docs:

        source = doc.metadata.get(

            "source",

            "Unknown"
        )

        page = doc.metadata.get(

            "page",

            "N/A"
        )

        source_text = (

            f"{source} — Page {page}"
        )

        if source_text not in shown_sources:

            formatted_sources.append(

                source_text
            )

            shown_sources.add(

                source_text
            )

    return formatted_sources