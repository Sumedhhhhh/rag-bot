from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.utils.logger import logger

def split_documents(documents):
    logger.info(f"Splitting {len(documents)} page(s) with fixed-size chunker (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    logger.info(f"Fixed-size chunking done: {len(chunks)} chunks")
    return chunks

def split_documents_semantic(documents, embedding_model):
    logger.info(f"Splitting {len(documents)} page(s) with SemanticChunker")
    splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile")
    chunks = splitter.split_documents(documents)
    logger.info(f"Semantic chunking done: {len(chunks)} chunks")
    return chunks