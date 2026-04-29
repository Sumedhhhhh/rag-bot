from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil

from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents, split_documents_semantic
from app.ingestion.json_doc_loader import load_json_doc
from app.embeddings.embedder import get_embedding_model
from app.vectorstore.store import build_vectorstore
from app.utils.logger import logger

router = APIRouter()

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    file_path = f"data/{file.filename}"

    logger.info(f"Received file: {file.filename}")

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    logger.info(f"File saved to {file_path}")

    embedding_model = get_embedding_model()

    if file.filename.endswith(".pdf"):
        logger.info("Loading PDF document")
        docs = load_pdf(file_path)
        logger.info(f"PDF loaded: {len(docs)} page(s)")
        chunks = split_documents_semantic(docs, embedding_model)
    elif file.filename.endswith(".json"):
        logger.info("Loading JSON document")
        docs = load_json_doc(file_path)
        chunks = docs
        logger.info(f"JSON loaded and flattened: {len(chunks)} fact(s), skipping splitter")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    build_vectorstore(chunks, embedding_model)

    logger.info(f"Document '{file.filename}' indexed successfully")

    return {"message": "Document uploaded and indexed successfully"}
