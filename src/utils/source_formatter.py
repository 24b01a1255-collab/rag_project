def show_sources(

    docs,

    st
):

    st.markdown("### Sources")

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

            st.markdown(

                f"- {source_text}"
            )

            shown_sources.add(

                source_text
            )