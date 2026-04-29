from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil

from app.qa.pipeline import run_pipeline
from app.utils.logger import logger

router = APIRouter()

@router.post("/infer")
async def infer(
    document: UploadFile = File(...),
    questions: UploadFile = File(...),
):
    if not (document.filename.endswith(".pdf") or document.filename.endswith(".json")):
        raise HTTPException(status_code=400, detail="Document must be a PDF or JSON file")

    if not questions.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Questions file must be JSON")

    doc_path = f"data/{document.filename}"
    questions_path = f"data/{questions.filename}"

    with open(doc_path, "wb") as f:
        shutil.copyfileobj(document.file, f)

    with open(questions_path, "wb") as f:
        shutil.copyfileobj(questions.file, f)

    logger.info(f"Starting /infer | document: {document.filename} | questions: {questions.filename}")

    try:
        answers = run_pipeline(doc_path, questions_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"/infer complete: {len(answers)} answer(s) returned")
    return answers
