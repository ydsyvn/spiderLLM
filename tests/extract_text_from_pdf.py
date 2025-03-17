import PyPDF2


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

celtic_knots_pdf = 'knot theory/celtic_knots.pdf'
extracted_text = extract_text_from_pdf(celtic_knots_pdf)

print(extracted_text)
