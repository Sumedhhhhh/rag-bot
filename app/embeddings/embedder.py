from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from app.config import EMBEDDING_MODEL, USE_OPENAI, OPENAI_API_KEY

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        if USE_OPENAI:
            _model = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        else:
            _model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _model