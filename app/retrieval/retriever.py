from app.config import TOP_K
from app.utils.logger import logger

def retrieve(vectorstore, query):
    logger.info(f"Retrieving top {TOP_K} chunks for query: '{query}'")
    docs = vectorstore.similarity_search(query, k=TOP_K)
    logger.info(f"Retrieved {len(docs)} chunks")
    return docs