import tempfile

from langchain_community.document_loaders import (
    PyPDFLoader
)

from langchain_core.documents import Document

from pdf2image import convert_from_path

import pytesseract


def extract_text_using_ocr(pdf_path, file_name):

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

    return documents


def load_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(

        delete=False,

        suffix=".pdf"

    ) as tmp_file:

        tmp_file.write(uploaded_file.read())

        temp_path = tmp_file.name

    loader = PyPDFLoader(temp_path)

    documents = loader.load()

    extracted_text = ""

    for doc in documents:

        extracted_text += doc.page_content.strip()

    # OCR fallback

    if len(extracted_text) < 50:

        documents = extract_text_using_ocr(

            temp_path,

            uploaded_file.name
        )

    else:

        for doc in documents:

            doc.metadata["source"] = uploaded_file.name

    return documents