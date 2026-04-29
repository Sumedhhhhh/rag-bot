import io
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def mock_doc(content="test content"):
    doc = MagicMock()
    doc.page_content = content
    return doc


def test_upload_unsupported_file_type_returns_400():
    response = client.post(
        "/upload",
        files={"file": ("report.txt", io.BytesIO(b"hello"), "text/plain")}
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_upload_json_returns_200():
    with patch("app.api.routes.upload.load_json_doc", return_value=[mock_doc()]), \
         patch("app.api.routes.upload.get_embedding_model", return_value=MagicMock()), \
         patch("app.api.routes.upload.build_vectorstore", return_value=MagicMock()), \
         patch("app.api.routes.upload.shutil.copyfileobj"):
        response = client.post(
            "/upload",
            files={"file": ("test_doc.json", io.BytesIO(b'{"key": "value"}'), "application/json")}
        )
    assert response.status_code == 200
    assert response.json()["message"] == "Document uploaded and indexed successfully"


def test_upload_pdf_returns_200():
    with patch("app.api.routes.upload.load_pdf", return_value=[mock_doc()]), \
         patch("app.api.routes.upload.split_documents_semantic", return_value=[mock_doc()]), \
         patch("app.api.routes.upload.get_embedding_model", return_value=MagicMock()), \
         patch("app.api.routes.upload.build_vectorstore", return_value=MagicMock()), \
         patch("app.api.routes.upload.shutil.copyfileobj"):
        response = client.post(
            "/upload",
            files={"file": ("test_doc.pdf", io.BytesIO(b"%PDF content"), "application/pdf")}
        )
    assert response.status_code == 200


def test_upload_pdf_calls_semantic_chunker():
    with patch("app.api.routes.upload.load_pdf", return_value=[mock_doc()]), \
         patch("app.api.routes.upload.split_documents_semantic", return_value=[mock_doc()]) as mock_split, \
         patch("app.api.routes.upload.get_embedding_model", return_value=MagicMock()), \
         patch("app.api.routes.upload.build_vectorstore", return_value=MagicMock()), \
         patch("app.api.routes.upload.shutil.copyfileobj"):
        client.post(
            "/upload",
            files={"file": ("test_doc.pdf", io.BytesIO(b"%PDF content"), "application/pdf")}
        )
    mock_split.assert_called_once()


def test_upload_json_skips_splitter():
    with patch("app.api.routes.upload.load_json_doc", return_value=[mock_doc()]), \
         patch("app.api.routes.upload.split_documents_semantic") as mock_split, \
         patch("app.api.routes.upload.get_embedding_model", return_value=MagicMock()), \
         patch("app.api.routes.upload.build_vectorstore", return_value=MagicMock()), \
         patch("app.api.routes.upload.shutil.copyfileobj"):
        client.post(
            "/upload",
            files={"file": ("test_doc.json", io.BytesIO(b'{"k":"v"}'), "application/json")}
        )
    mock_split.assert_not_called()


def test_upload_calls_build_vectorstore():
    with patch("app.api.routes.upload.load_json_doc", return_value=[mock_doc()]), \
         patch("app.api.routes.upload.get_embedding_model", return_value=MagicMock()), \
         patch("app.api.routes.upload.build_vectorstore", return_value=MagicMock()) as mock_build, \
         patch("app.api.routes.upload.shutil.copyfileobj"):
        client.post(
            "/upload",
            files={"file": ("test_doc.json", io.BytesIO(b'{"k":"v"}'), "application/json")}
        )
    mock_build.assert_called_once()
