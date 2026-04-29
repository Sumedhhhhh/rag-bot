from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def mock_doc(content="The SLA is 72 hours."):
    doc = MagicMock()
    doc.page_content = content
    return doc


def test_ask_empty_questions_returns_400():
    response = client.post("/ask", json={"questions": []})
    assert response.status_code == 400
    assert "cannot be empty" in response.json()["detail"]


def test_ask_without_uploaded_document_returns_400():
    with patch("app.api.routes.qa.get_embedding_model", return_value=MagicMock()), \
         patch("app.api.routes.qa.load_vectorstore", side_effect=Exception("Vectorstore not found")):
        response = client.post("/ask", json={"questions": ["What is the SLA?"]})
    assert response.status_code == 400


def test_ask_returns_question_answer_dict():
    with patch("app.api.routes.qa.get_embedding_model", return_value=MagicMock()), \
         patch("app.api.routes.qa.load_vectorstore", return_value=MagicMock()), \
         patch("app.api.routes.qa.retrieve", return_value=[mock_doc()]), \
         patch("app.api.routes.qa.generate_answer_openai", return_value="SLA is 72 hours."):
        response = client.post("/ask", json={"questions": ["What is the SLA?"]})
    assert response.status_code == 200
    data = response.json()
    assert "What is the SLA?" in data
    assert data["What is the SLA?"] == "SLA is 72 hours."


def test_ask_multiple_questions_returns_all_answers():
    with patch("app.api.routes.qa.get_embedding_model", return_value=MagicMock()), \
         patch("app.api.routes.qa.load_vectorstore", return_value=MagicMock()), \
         patch("app.api.routes.qa.retrieve", return_value=[mock_doc()]), \
         patch("app.api.routes.qa.generate_answer_openai", return_value="Some answer"):
        response = client.post("/ask", json={"questions": ["Q1?", "Q2?", "Q3?"]})
    assert response.status_code == 200
    assert len(response.json()) == 3


def test_ask_response_keys_match_questions():
    questions = ["What is the SLA?", "Who is the CEO?"]
    with patch("app.api.routes.qa.get_embedding_model", return_value=MagicMock()), \
         patch("app.api.routes.qa.load_vectorstore", return_value=MagicMock()), \
         patch("app.api.routes.qa.retrieve", return_value=[mock_doc()]), \
         patch("app.api.routes.qa.generate_answer_openai", return_value="Answer"):
        response = client.post("/ask", json={"questions": questions})
    assert set(response.json().keys()) == set(questions)


def test_ask_calls_retrieve_once_per_question():
    with patch("app.api.routes.qa.get_embedding_model", return_value=MagicMock()), \
         patch("app.api.routes.qa.load_vectorstore", return_value=MagicMock()), \
         patch("app.api.routes.qa.retrieve", return_value=[mock_doc()]) as mock_retrieve, \
         patch("app.api.routes.qa.generate_answer_openai", return_value="Answer"):
        client.post("/ask", json={"questions": ["Q1?", "Q2?", "Q3?"]})
    assert mock_retrieve.call_count == 3
