import os
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

from app.utils.document_loader import get_loader, clean_text, process_documents
from langchain_core.documents import Document


def test_clean_text():
    text = "Hello\x00World"
    cleaned = clean_text(text)
    assert "\x00" not in cleaned
    assert cleaned == "HelloWorld"


def test_get_loader_text(tmp_path):
    # Create a temporary text file.
    file_path = tmp_path / "test.txt"
    file_path.write_text("Sample text")
    loader, known_type, file_ext = get_loader("test.txt", "text/plain", str(file_path))
    assert known_type is True
    assert file_ext == "txt"
    data = loader.load()
    # Check that data is loaded.
    assert data is not None


def test_process_documents():
    docs = [
        Document(
            page_content="Page 1 content", metadata={"source": "dummy.txt", "page": 1}
        ),
        Document(
            page_content="Page 2 content", metadata={"source": "dummy.txt", "page": 2}
        ),
    ]
    processed = process_documents(docs)
    assert "dummy.txt" in processed
    assert "# PAGE 1" in processed
    assert "# PAGE 2" in processed


def test_safe_pdf_loader_class():
    """Test that SafePyPDFLoader class can be instantiated"""
    from app.utils.document_loader import SafePyPDFLoader

    # Test instantiation
    loader = SafePyPDFLoader("dummy.pdf", extract_images=True)
    assert loader.filepath == "dummy.pdf"
    assert loader.extract_images == True
    assert loader._temp_filepath is None


def test_get_loader_text_lazy_load(tmp_path):
    """Test that lazy_load returns an iterator yielding documents."""
    file_path = tmp_path / "test.txt"
    file_path.write_text("Sample text")
    loader, known_type, file_ext = get_loader("test.txt", "text/plain", str(file_path))
    assert known_type is True
    assert file_ext == "txt"
    data = list(loader.lazy_load())
    assert len(data) > 0
    assert hasattr(data[0], "page_content")


def test_get_loader_pdf(tmp_path):
    """Test get_loader returns SafePyPDFLoader for PDF files"""
    # Create a dummy PDF file path (doesn't need to be real for this test)
    file_path = tmp_path / "test.pdf"
    file_path.write_text("dummy content")  # Not a real PDF, but that's OK for this test

    loader, known_type, file_ext = get_loader(
        "test.pdf", "application/pdf", str(file_path)
    )

    # Check that we get our SafePyPDFLoader
    from app.utils.document_loader import SafePyPDFLoader

    assert isinstance(loader, SafePyPDFLoader)
    assert known_type is True
    assert file_ext == "pdf"


def test_safe_pdf_loader_lazy_load():
    """Test that SafePyPDFLoader.lazy_load() returns an Iterator."""
    from app.utils.document_loader import SafePyPDFLoader

    loader = SafePyPDFLoader("dummy.pdf", extract_images=False)
    assert hasattr(loader, "lazy_load")
    result = loader.lazy_load()
    assert isinstance(result, Iterator)


def test_safe_pdf_loader_fallback_no_duplicate_pages():
    """Fallback after mid-stream KeyError must not duplicate already-yielded pages."""
    from app.utils.document_loader import SafePyPDFLoader

    fallback_docs = [Document(page_content=f"fallback page {i}") for i in range(5)]

    def primary_gen():
        yield Document(page_content="partial page 0")
        yield Document(page_content="partial page 1")
        raise KeyError("/Filter")

    def fallback_gen():
        yield from fallback_docs

    loader = SafePyPDFLoader("dummy.pdf", extract_images=True)

    with patch("app.utils.document_loader.PyPDFLoader") as MockPDF:
        primary_instance = MagicMock()
        primary_instance.lazy_load.side_effect = primary_gen
        fallback_instance = MagicMock()
        fallback_instance.lazy_load.side_effect = fallback_gen
        MockPDF.side_effect = [primary_instance, fallback_instance]

        result = list(loader.lazy_load())

    # Must be exactly the 5 fallback pages, NOT 2 partial + 5 fallback = 7
    assert len(result) == 5
    assert result[0].page_content == "fallback page 0"
    assert result[-1].page_content == "fallback page 4"


def test_safe_pdf_loader_fallback_via_load():
    """load() delegates to lazy_load(), so fallback must also be correct via load()."""
    from app.utils.document_loader import SafePyPDFLoader

    fallback_docs = [Document(page_content=f"fb {i}") for i in range(3)]

    def primary_gen():
        yield Document(page_content="partial 0")
        raise KeyError("/Filter")

    def fallback_gen():
        yield from fallback_docs

    loader = SafePyPDFLoader("dummy.pdf", extract_images=True)

    with patch("app.utils.document_loader.PyPDFLoader") as MockPDF:
        primary_instance = MagicMock()
        primary_instance.lazy_load.side_effect = primary_gen
        fallback_instance = MagicMock()
        fallback_instance.lazy_load.side_effect = fallback_gen
        MockPDF.side_effect = [primary_instance, fallback_instance]

        result = loader.load()

    assert len(result) == 3
    assert result[0].page_content == "fb 0"


def test_safe_pdf_loader_non_filter_error_propagates():
    """KeyError that isn't /Filter should propagate, not silently fallback."""
    from app.utils.document_loader import SafePyPDFLoader
    import pytest

    def bad_gen():
        raise KeyError("SomeOtherKey")

    loader = SafePyPDFLoader("dummy.pdf", extract_images=True)

    with patch("app.utils.document_loader.PyPDFLoader") as MockPDF:
        instance = MagicMock()
        instance.lazy_load.side_effect = bad_gen
        MockPDF.return_value = instance

        with pytest.raises(KeyError, match="SomeOtherKey"):
            list(loader.lazy_load())
