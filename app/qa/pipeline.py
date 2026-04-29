from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents
from app.embeddings.embedder import get_embedding_model
from app.vectorstore.store import build_vectorstore
from app.retrieval.retriever import retrieve
from app.ingestion.json_loader import load_questions
from app.qa.mock_llm import generate_answer
from app.qa.llm import generate_answer_ollama
from app.qa.openai import generate_answer_openai
from app.config import USE_OLLAMA, USE_OPENAI
from app.utils.logger import logger


def run_pipeline(pdf_path, question_path) :
    docs = load_pdf(pdf_path)
    logger.info("Starting document processing")
    
    chunks = split_documents(docs)
    logger.info(f"Total chunks created: {len(chunks)}")
    
    embedding_model = get_embedding_model()
    
    vector_store = build_vectorstore(chunks, embedding_model)
    questions = load_questions(question_path)

    answers = {}

    for q in questions :
        retrieved_docs = retrieve(vector_store,q)
        if USE_OPENAI:
            answer = generate_answer_openai(q, retrieved_docs)
        elif USE_OLLAMA:
            answer = generate_answer_ollama(q, retrieved_docs)
        else:
            answer = generate_answer(q, retrieved_docs)

        answers[q] = answer

    return answers