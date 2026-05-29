import pandas as pd

from langchain_core.documents import Document


def load_csv(uploaded_file):

    df = pd.read_csv(uploaded_file)

    text = df.to_string()

    documents = [

        Document(

            page_content=text,

            metadata={

                "source": uploaded_file.name,

                "page": 1
            }
        )
    ]

    return documents