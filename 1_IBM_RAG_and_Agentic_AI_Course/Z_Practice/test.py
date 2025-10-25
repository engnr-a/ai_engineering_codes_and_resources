pdf_document_urls =[
    "https://group.dhl.com/content/dam/deutschepostdhl/de/media-center/investors/documents/business-profiles/DHL-Group-2025-Business-Profile.pdf",
    "https://group.dhl.com/content/dam/deutschepostdhl/en/media-center/responsibility/dhl-group-code-of-conduct-en.pdf",
    "https://group.dhl.com/content/dam/deutschepostdhl/en/media-center/responsibility/dhl-group-health-and-wellbeing-policy.pdf",
    "https://group.dhl.com/content/dam/deutschepostdhl/en/media-center/responsibility/occupational-health-and-safety-policy-statement.pdf"
]
# loader = PyPDFLoader(pdf_document_urls[0]).load()
all_loaded_documents = []
for url in pdf_document_urls:
    loader = PyPDFLoader(url)
    all_loaded_documents.extend(loader.load())
    