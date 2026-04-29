from app.ingestion.loader import load_pdf
from app.ingestion.json_doc_loader import load_json_doc
from app.ingestion.splitter import split_documents_semantic
from app.embeddings.embedder import get_embedding_model
from app.vectorstore.store import build_vectorstore
from app.retrieval.retriever import retrieve
from app.ingestion.json_loader import load_questions
from app.qa.mock_llm import generate_answer
from app.qa.llm import generate_answer_ollama
from app.qa.openai import generate_answer_openai
from app.config import USE_OLLAMA, USE_OPENAI
from app.utils.logger import logger


def run_pipeline(doc_path, question_path):
    logger.info(f"Pipeline started | document: {doc_path} | questions: {question_path}")

    embedding_model = get_embedding_model()

    if doc_path.endswith(".pdf"):
        logger.info("Loading PDF document")
        docs = load_pdf(doc_path)
        logger.info(f"PDF loaded: {len(docs)} page(s)")
        chunks = split_documents_semantic(docs, embedding_model)
    elif doc_path.endswith(".json"):
        logger.info("Loading JSON document")
        docs = load_json_doc(doc_path)
        chunks = docs
        logger.info(f"JSON loaded and flattened: {len(chunks)} fact(s), skipping splitter")
    else:
        raise ValueError(f"Unsupported file type: {doc_path}")

    vector_store = build_vectorstore(chunks, embedding_model)

    questions = load_questions(question_path)
    logger.info(f"Loaded {len(questions)} question(s) from {question_path}")

    answers = {}

    for i, q in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] Answering: '{q}'")
        retrieved_docs = retrieve(vector_store, q)
        if USE_OPENAI:
            answer = generate_answer_openai(q, retrieved_docs)
        elif USE_OLLAMA:
            answer = generate_answer_ollama(q, retrieved_docs)
        else:
            answer = generate_answer(q, retrieved_docs)

        answers[q] = answer

    logger.info(f"Pipeline complete: {len(answers)} answer(s) generated")
    return answers