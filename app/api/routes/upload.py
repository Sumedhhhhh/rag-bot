from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil

from app.ingestion.loader import load_pdf
from app.ingestion.splitter import split_documents
from app.ingestion.json_doc_loader import load_json_doc
from app.embeddings.embedder import get_embedding_model
from app.vectorstore.store import build_vectorstore
from app.utils.logger import logger

router = APIRouter()

@router.post("/upload")
async def upload_file( file : UploadFile = File(...)) :

    file_path = f"data/{file.filename}"

 

    logger.info(f"Uploading file: {file.filename}")

    with open(file_path, "wb") as buffer :
        shutil.copyfileobj(file.file, buffer)

    if file.filename.endswith(".pdf") :
        docs = load_pdf(file_path)
    elif file.filename.endswith(".json") :
        docs = load_json_doc(file_path)
    else :
        raise HTTPException(status_code=400, detail="Unsupported file type")

    
    chunks = split_documents(docs)

    embedding_model = get_embedding_model()
    build_vectorstore(chunks, embedding_model)

    logger.info("Document indexed successfully")

    return { "message" : "Document uploaded and indexed successfully" }
