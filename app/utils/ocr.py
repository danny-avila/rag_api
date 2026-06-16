# app/utils/ocr.py
"""OCR fallback for text-less PDFs (scanned pages / CAD exports).

Many CAD-export and scanned PDFs have no text layer, so ``pypdf.extract_text()``
returns an empty string and the document silently ingests as zero chunks. This
module provides:

  - :func:`has_text_layer` to detect that situation, and
  - :func:`ocr_pdf` to recover text by rasterizing each page with ``pypdfium2``
    and running OCR with ``rapidocr-onnxruntime`` (already a project dependency).

The OCR engine is injectable so the bulk of the logic is unit-testable without
loading an ONNX model.
"""

import logging
from typing import Callable, List, Optional, Sequence

import numpy as np
import pypdfium2 as pdfium
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Minimum number of non-whitespace characters across all pages for a PDF to be
# considered as having a usable text layer. Below this we treat extraction as
# empty/garbage and fall back to OCR.
DEFAULT_OCR_MIN_CHARS = 16
# Render resolution for rasterizing pages before OCR.
DEFAULT_OCR_DPI = 200
# Hard cap on the number of pages OCR'd, to guard against runaway cost on very
# large documents.
DEFAULT_OCR_MAX_PAGES = 50

# An OCR callable maps a rendered page image (H x W x 3 uint8 array) to text.
OcrCallable = Callable[[np.ndarray], str]


class NoExtractableTextError(Exception):
    """Raised when a PDF yields no usable text even after the OCR fallback.

    Routes translate this into a clear 4xx response instead of a generic
    failure or a silent zero-chunk "success".
    """


def has_text_layer(
    pages: Sequence[Document], min_chars: int = DEFAULT_OCR_MIN_CHARS
) -> bool:
    """Return True if ``pages`` collectively contain a usable text layer.

    Counts non-whitespace characters across every page's ``page_content`` and
    short-circuits as soon as the threshold is met.
    """
    total = 0
    for page in pages:
        content = getattr(page, "page_content", "") or ""
        total += sum(1 for char in content if not char.isspace())
        if total >= min_chars:
            return True
    return False


def _ocr_result_to_text(result) -> str:
    """Defensively turn a rapidocr result into text.

    rapidocr returns either ``None`` (nothing found) or a list of rows shaped
    ``(box, text, confidence)``. We tolerate unexpected/empty rows rather than
    letting an ``IndexError`` escape and be reported as a generic failure.
    """
    if not result:
        return ""
    texts: List[str] = []
    for row in result:
        if isinstance(row, (list, tuple)) and len(row) >= 2 and row[1]:
            texts.append(str(row[1]))
    return "\n".join(texts)


def _build_default_ocr() -> OcrCallable:
    """Construct the default rapidocr-backed OCR callable.

    Imported lazily so that importing this module (and the fast unit tests that
    inject a fake OCR callable) does not pull in the ONNX runtime.
    """
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()

    def run(image: np.ndarray) -> str:
        result, _elapsed = engine(image)
        return _ocr_result_to_text(result)

    return run


def _render_page(page, dpi: int) -> np.ndarray:
    """Rasterize a single pypdfium2 page to an RGB numpy array."""
    bitmap = page.render(scale=dpi / 72.0)
    pil_image = bitmap.to_pil().convert("RGB")
    return np.asarray(pil_image)


def ocr_pdf(
    filepath: str,
    *,
    dpi: int = DEFAULT_OCR_DPI,
    max_pages: int = DEFAULT_OCR_MAX_PAGES,
    ocr: Optional[OcrCallable] = None,
) -> List[Document]:
    """OCR every page of ``filepath`` and return one Document per page.

    :param ocr: optional injected OCR callable (defaults to rapidocr).
    :returns: a list of Documents with ``ocr=True`` and a 0-based ``page`` in
        their metadata. The text may be empty for a page if OCR found nothing.
    """
    if ocr is None:
        ocr = _build_default_ocr()

    pdf = pdfium.PdfDocument(filepath)
    try:
        total_pages = len(pdf)
        page_count = min(total_pages, max_pages)
        if total_pages > max_pages:
            logger.warning(
                "OCR truncated '%s': %d pages in document, limit is %d "
                "(raise RAG_PDF_OCR_MAX_PAGES to process the full document).",
                filepath,
                total_pages,
                max_pages,
            )
        documents: List[Document] = []
        for index in range(page_count):
            page = pdf[index]
            try:
                image = _render_page(page, dpi)
                text = ocr(image) or ""
            finally:
                page.close()
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(filepath), "page": index, "ocr": True},
                )
            )
        return documents
    finally:
        pdf.close()
