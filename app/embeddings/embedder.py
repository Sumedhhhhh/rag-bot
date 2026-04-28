from langchain_huggingface import HuggingFaceEmbeddings
from app.config import EMBEDDING_MODEL

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _model