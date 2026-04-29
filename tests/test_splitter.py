from langchain.schema import Document
from app.ingestion.splitter import split_documents


def make_doc(content):
    return Document(page_content=content)


def test_long_document_is_split_into_multiple_chunks():
    long_text = "This is a sentence about security policies. " * 50
    chunks = split_documents([make_doc(long_text)])
    assert len(chunks) > 1


def test_short_document_stays_as_single_chunk():
    chunks = split_documents([make_doc("Short text.")])
    assert len(chunks) == 1


def test_chunks_do_not_exceed_max_size():
    long_text = "word " * 500
    chunks = split_documents([make_doc(long_text)])
    for chunk in chunks:
        assert len(chunk.page_content) <= 700  # CHUNK_SIZE=600 + overlap buffer


def test_all_chunks_have_page_content():
    chunks = split_documents([make_doc("Some content here.")])
    assert all(hasattr(c, "page_content") for c in chunks)
    assert all(len(c.page_content) > 0 for c in chunks)


def test_empty_list_returns_empty():
    assert split_documents([]) == []


def test_multiple_documents_are_all_chunked():
    docs = [make_doc("Document one content. " * 40),
            make_doc("Document two content. " * 40)]
    chunks = split_documents(docs)
    assert len(chunks) > 2
