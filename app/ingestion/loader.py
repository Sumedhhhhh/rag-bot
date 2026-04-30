def load_pdf(path: str):
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(path)
    return loader.load()
