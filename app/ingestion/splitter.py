from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

def split_documents_semantic(documents, embedding_model):
    splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile")
    return splitter.split_documents(documents)