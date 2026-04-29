from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from app.config import EMBEDDING_MODEL, USE_OPENAI, OPENAI_API_KEY
from app.utils.logger import logger

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        if USE_OPENAI:
            logger.info(f"Loading OpenAI embedding model: {EMBEDDING_MODEL}")
            _model = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        else:
            logger.info(f"Loading HuggingFace embedding model: {EMBEDDING_MODEL}")
            _model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info("Embedding model ready")
    else:
        logger.debug("Embedding model already loaded, reusing cached instance")
    return _model