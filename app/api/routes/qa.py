from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.embeddings.embedder import get_embedding_model
from app.vectorstore.store import load_vectorstore
from app.retrieval.retriever import retrieve
from app.qa.mock_llm import generate_answer
from app.qa.llm import generate_answer_ollama
from app.qa.openai import generate_answer_openai
from app.config import USE_OLLAMA, USE_OPENAI
from app.utils.logger import logger

router = APIRouter()

class QuestionRequest(BaseModel) :
    questions : list[str]

@router.post("/ask")
def ask_questions(request : QuestionRequest) :
    

    if not request.questions:
        raise HTTPException(status_code=400, detail="Questions list cannot be empty")

    try:
        embedding_model = get_embedding_model()
        vector_store = load_vectorstore(embedding_model)
    except Exception:
        raise HTTPException(status_code=400, detail="Please upload a document first")

    answers = {}

    total = len(request.questions)
    for i, q in enumerate(request.questions, 1):
        logger.info(f"[{i}/{total}] Processing: '{q}'")
        docs = retrieve(vector_store, q)
        logger.debug(f"Question: {q} | chunks: {[d.page_content for d in docs]}")

        if USE_OPENAI:
            answer = generate_answer_openai(q, docs)
        elif USE_OLLAMA:
            answer = generate_answer_ollama(q, docs)
        else:
            answer = generate_answer(q, docs)

        answers[q] = answer

    return answers