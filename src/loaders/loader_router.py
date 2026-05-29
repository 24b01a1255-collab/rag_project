from src.loaders.pdf_loader import load_pdf

from src.loaders.docx_loader import load_docx

from src.loaders.csv_loader import load_csv

from src.loaders.pptx_loader import load_pptx


def load_document(uploaded_file):

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):

        return load_pdf(uploaded_file)

    elif file_name.endswith(".docx"):

        return load_docx(uploaded_file)

    elif file_name.endswith(".csv"):

        return load_csv(uploaded_file)

    elif file_name.endswith(".pptx"):

        return load_pptx(uploaded_file)

    else:

        return []