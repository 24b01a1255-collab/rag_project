from duckduckgo_search import DDGS


def web_search(query, max_results=5):

    results_text = ""

    try:

        with DDGS() as ddgs:

            results = ddgs.text(
                query,
                max_results=max_results
            )

            for result in results:

                results_text += (
                    f"Title: {result['title']}\n"
                    f"Body: {result['body']}\n"
                    f"Link: {result['href']}\n\n"
                )

    except Exception as e:

        results_text = f"Web search failed: {str(e)}"

    return results_text