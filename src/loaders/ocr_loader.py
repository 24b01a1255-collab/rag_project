from pdf2image import convert_from_path

import pytesseract

from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyPDFLoader
)

def extract_documents(

    pdf_path,

    file_name
):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    extracted_text = ""

    for doc in documents:

        extracted_text += doc.page_content.strip()

    # ----------------------------------------
    # OCR FALLBACK
    # ----------------------------------------

    if len(extracted_text) < 50:

        documents = []

        images = convert_from_path(pdf_path)

        for i, image in enumerate(images):

            text = pytesseract.image_to_string(image)

            if text.strip():

                doc = Document(

                    page_content=text,

                    metadata={

                        "source": file_name,

                        "page": i + 1
                    }
                )

                documents.append(doc)

    else:

        for doc in documents:

            doc.metadata["source"] = file_name

    return documents