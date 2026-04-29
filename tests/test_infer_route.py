import io
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

QUESTIONS_JSON = b'{"questions": ["What is the SLA?"]}'
PDF_CONTENT    = b"%PDF-1.4 test content"
JSON_CONTENT   = b'{"company": {"name": "Acme"}}'


def test_infer_unsupported_document_type_returns_400():
    response = client.post(
        "/infer",
        files={
            "document":  ("doc.txt",  io.BytesIO(b"hello"),         "text/plain"),
            "questions": ("q.json",   io.BytesIO(QUESTIONS_JSON),   "application/json"),
        }
    )
    assert response.status_code == 400
    assert "PDF or JSON" in response.json()["detail"]


def test_infer_non_json_questions_file_returns_400():
    response = client.post(
        "/infer",
        files={
            "document":  ("doc.pdf",  io.BytesIO(PDF_CONTENT), "application/pdf"),
            "questions": ("q.txt",    io.BytesIO(b"questions"), "text/plain"),
        }
    )
    assert response.status_code == 400
    assert "JSON" in response.json()["detail"]


def test_infer_pdf_document_returns_200():
    mock_answers = {"What is the SLA?": "Clients are notified within 72 hours."}
    with patch("app.api.routes.infer.run_pipeline", return_value=mock_answers), \
         patch("app.api.routes.infer.shutil.copyfileobj"):
        response = client.post(
            "/infer",
            files={
                "document":  ("test_doc.pdf", io.BytesIO(PDF_CONTENT),  "application/pdf"),
                "questions": ("test_q.json",  io.BytesIO(QUESTIONS_JSON), "application/json"),
            }
        )
    assert response.status_code == 200
    assert response.json() == mock_answers


def test_infer_json_document_returns_200():
    mock_answers = {"What is the company name?": "Acme"}
    with patch("app.api.routes.infer.run_pipeline", return_value=mock_answers), \
         patch("app.api.routes.infer.shutil.copyfileobj"):
        response = client.post(
            "/infer",
            files={
                "document":  ("test_doc.json", io.BytesIO(JSON_CONTENT),    "application/json"),
                "questions": ("test_q.json",   io.BytesIO(QUESTIONS_JSON),  "application/json"),
            }
        )
    assert response.status_code == 200
    assert response.json() == mock_answers


def test_infer_calls_run_pipeline_with_correct_paths():
    with patch("app.api.routes.infer.run_pipeline", return_value={}) as mock_pipeline, \
         patch("app.api.routes.infer.shutil.copyfileobj"):
        client.post(
            "/infer",
            files={
                "document":  ("test_doc.pdf",  io.BytesIO(PDF_CONTENT),    "application/pdf"),
                "questions": ("test_q.json",   io.BytesIO(QUESTIONS_JSON), "application/json"),
            }
        )
    mock_pipeline.assert_called_once_with(
        "data/test_doc.pdf",
        "data/test_q.json"
    )


def test_infer_pipeline_error_returns_500():
    with patch("app.api.routes.infer.run_pipeline", side_effect=Exception("Pipeline failed")), \
         patch("app.api.routes.infer.shutil.copyfileobj"):
        response = client.post(
            "/infer",
            files={
                "document":  ("test_doc.pdf", io.BytesIO(PDF_CONTENT),    "application/pdf"),
                "questions": ("test_q.json",  io.BytesIO(QUESTIONS_JSON), "application/json"),
            }
        )
    assert response.status_code == 500
    assert "Pipeline failed" in response.json()["detail"]
