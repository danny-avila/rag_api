from parsers import clean_text, process_documents
from langchain.schema import Document

def test_clean_text():
    raw_text = "Hello\x00World"
    assert "\x00" in raw_text  # Verify raw text contains null byte
    cleaned = clean_text(raw_text)
    assert "\x00" not in cleaned

def test_process_documents():
    # Create dummy documents with page metadata.
    doc1 = Document(page_content="Content 1", metadata={"page": 1, "source": "file1.pdf"})
    doc2 = Document(page_content="Content 2", metadata={"page": 2})
    result = process_documents([doc1, doc2])
    assert "file1.pdf" in result
    assert "# PAGE 1" in result
    assert "# PAGE 2" in result