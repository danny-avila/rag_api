# app/utils/document_loader.py
import os
import codecs
import tempfile

from typing import List, Optional

from langchain_core.documents import Document

from app.config import known_source_ext, PDF_EXTRACT_IMAGES, CHUNK_OVERLAP, logger
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    UnstructuredEPubLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredRSTLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)


def detect_file_encoding(filepath: str) -> str:
    """
    Detect the encoding of a file by checking for BOM markers.
    Returns the detected encoding or 'utf-8' as default.
    """
    with open(filepath, "rb") as f:
        raw = f.read(4)

    # Check for BOM markers
    if raw.startswith(codecs.BOM_UTF16_LE):
        return "utf-16-le"
    elif raw.startswith(codecs.BOM_UTF16_BE):
        return "utf-16-be"
    elif raw.startswith(codecs.BOM_UTF16):
        return "utf-16"
    elif raw.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    elif raw.startswith(codecs.BOM_UTF32_LE):
        return "utf-32-le"
    elif raw.startswith(codecs.BOM_UTF32_BE):
        return "utf-32-be"
    else:
        # Default to utf-8 if no BOM is found
        return "utf-8"


def cleanup_temp_encoding_file(loader) -> None:
    """
    Clean up temporary UTF-8 file if it was created for encoding conversion.

    :param loader: The document loader that may have created a temporary file
    """
    if hasattr(loader, "_temp_filepath"):
        try:
            os.remove(loader._temp_filepath)
        except Exception as e:
            logger.warning(f"Failed to remove temporary UTF-8 file: {e}")


def get_loader(filename: str, file_content_type: str, filepath: str):
    """Get the appropriate document loader based on file type and\or content type."""
    file_ext = filename.split(".")[-1].lower()
    known_type = True

    # File Content Type reference:
    # ref.: https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/MIME_types/Common_types
    if file_ext == "pdf" or file_content_type == "application/pdf":
        loader = SafePyPDFLoader(filepath, extract_images=PDF_EXTRACT_IMAGES)
    elif file_ext == "csv" or file_content_type == "text/csv":
        # Detect encoding for CSV files
        encoding = detect_file_encoding(filepath)

        if encoding != "utf-8":
            # For non-UTF-8 encodings, we need to convert the file first
            # Create a temporary UTF-8 file
            temp_file = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", suffix=".csv", delete=False
                ) as temp_file:
                    # Read the original file with detected encoding
                    with open(filepath, "r", encoding=encoding) as original_file:
                        content = original_file.read()
                        temp_file.write(content)

                    temp_filepath = temp_file.name

                # Use the temporary UTF-8 file with CSVLoader
                loader = CSVLoader(temp_filepath)

                # Store the temp file path for cleanup
                loader._temp_filepath = temp_filepath
            except Exception as e:
                # If temp file was created but there was an error, clean it up
                if temp_file and os.path.exists(temp_file.name):
                    os.unlink(temp_file.name)
                raise e
        else:
            loader = CSVLoader(filepath)
    elif file_ext == "rst":
        loader = UnstructuredRSTLoader(filepath, mode="elements")
    elif file_ext == "xml" or file_content_type in [
            "application/xml",
            "text/xml",
            "application/xhtml+xml",
        ]:
        loader = UnstructuredXMLLoader(filepath)
    elif file_ext in ["ppt", "pptx"] or file_content_type in [
            "application/vnd.ms-powerpoint",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ]:
        loader = UnstructuredPowerPointLoader(filepath)
    elif file_ext == "md" or file_content_type in [
            "text/markdown",
            "text/x-markdown",
            "application/markdown",
            "application/x-markdown",
        ]:
        loader = UnstructuredMarkdownLoader(filepath)
    elif file_ext == "epub" or file_content_type == "application/epub+zip":
        loader = UnstructuredEPubLoader(filepath)
    elif file_ext in ["doc", "docx"] or file_content_type in [
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]:
        loader = Docx2txtLoader(filepath)
    elif file_ext in ["xls", "xlsx"] or file_content_type in [
            "application/vnd.ms-excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ]:
        loader = UnstructuredExcelLoader(filepath)
    elif file_ext == "json" or file_content_type == "application/json":
        loader = TextLoader(filepath, autodetect_encoding=True)
    elif file_ext in known_source_ext or (
            file_content_type and file_content_type.find("text/") >= 0
        ):
        loader = TextLoader(filepath, autodetect_encoding=True)
    else:
        loader = TextLoader(filepath, autodetect_encoding=True)
        known_type = False

    return loader, known_type, file_ext


def clean_text(text: str) -> str:
    """
    Remove NUL (0x00) characters from a string.

    :param text: The original text with potential NUL characters.
    :return: Cleaned text without NUL characters.
    """
    return text.replace("\x00", "")


def process_documents(documents: List[Document]) -> str:
    processed_text = ""
    last_page: Optional[int] = None
    doc_basename = ""

    for doc in documents:
        if "source" in doc.metadata:
            doc_basename = doc.metadata["source"].split("/")[-1]
            break

    processed_text += f"{doc_basename}\n"

    for doc in documents:
        current_page = doc.metadata.get("page")
        if current_page and current_page != last_page:
            processed_text += f"\n# PAGE {doc.metadata['page']}\n\n"
            last_page = current_page

        new_content = doc.page_content
        if processed_text.endswith(new_content[:CHUNK_OVERLAP]):
            processed_text += new_content[CHUNK_OVERLAP:]
        else:
            processed_text += new_content

    return processed_text.strip()


class SafePyPDFLoader:
    """
    A wrapper around PyPDFLoader that handles image extraction failures gracefully.
    Falls back to text-only extraction when image extraction fails.

    This is a workaround for issues with PyPDFLoader that can occur when extracting images
    from PDFs, which can lead to KeyError exceptions if the PDF is malformed or has unsupported
    image formats. This class attempts to load the PDF with image extraction enabled, and if it
    fails due to a KeyError related to image filters, it falls back to loading the PDF
    without image extraction.
    ref.: https://github.com/langchain-ai/langchain/issues/26652
    """

    def __init__(self, filepath: str, extract_images: bool = False):
        self.filepath = filepath
        self.extract_images = extract_images
        self._temp_filepath = None  # For compatibility with cleanup function

    def load(self) -> List[Document]:
        """Load PDF documents with automatic fallback on image extraction errors."""
        loader = PyPDFLoader(self.filepath, extract_images=self.extract_images)

        try:
            return loader.load()
        except KeyError as e:
            if "/Filter" in str(e) and self.extract_images:
                logger.warning(
                    f"PDF image extraction failed for {self.filepath}, falling back to text-only: {e}"
                )
                fallback_loader = PyPDFLoader(self.filepath, extract_images=False)
                return fallback_loader.load()
            else:
                # Re-raise if it's a different error
                raise
