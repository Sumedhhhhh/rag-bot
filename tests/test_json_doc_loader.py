import json
import pytest
from app.ingestion.json_doc_loader import load_json_doc


def write_json(tmp_path, data):
    path = tmp_path / "test.json"
    path.write_text(json.dumps(data))
    return str(path)


def test_flat_json_creates_one_doc_per_key(tmp_path):
    path = write_json(tmp_path, {"name": "Alice", "age": 30})
    docs = load_json_doc(path)
    contents = [d.page_content for d in docs]
    assert "name: Alice" in contents
    assert "age: 30" in contents


def test_nested_json_uses_full_key_path(tmp_path):
    path = write_json(tmp_path, {"company": {"ceo": {"name": "Bob"}}})
    docs = load_json_doc(path)
    assert any("company ceo name: Bob" in d.page_content for d in docs)


def test_json_array_creates_one_doc_per_item(tmp_path):
    path = write_json(tmp_path, {"tags": ["security", "compliance", "cloud"]})
    docs = load_json_doc(path)
    assert len(docs) == 3


def test_deeply_nested_key_path_is_correct(tmp_path):
    path = write_json(tmp_path, {"a": {"b": {"c": {"d": "value"}}}})
    docs = load_json_doc(path)
    assert any("a b c d: value" in d.page_content for d in docs)


def test_all_docs_have_page_content(tmp_path):
    path = write_json(tmp_path, {"x": 1, "y": 2, "z": 3})
    docs = load_json_doc(path)
    assert all(hasattr(d, "page_content") for d in docs)
    assert all(len(d.page_content) > 0 for d in docs)


def test_numeric_values_are_stringified(tmp_path):
    path = write_json(tmp_path, {"count": 42, "ratio": 0.95})
    docs = load_json_doc(path)
    contents = [d.page_content for d in docs]
    assert any("42" in c for c in contents)
    assert any("0.95" in c for c in contents)


def test_mixed_nested_and_array(tmp_path):
    path = write_json(tmp_path, {
        "company": "Acme",
        "offices": ["NYC", "SF"],
        "ceo": {"name": "Alice", "age": 45}
    })
    docs = load_json_doc(path)
    contents = [d.page_content for d in docs]
    assert any("company: Acme" in c for c in contents)
    assert any("NYC" in c for c in contents)
    assert any("ceo name: Alice" in c for c in contents)
