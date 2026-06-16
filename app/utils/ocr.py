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

Two quality controls keep low-quality OCR out of the vector store and make the
pipeline observable:

  - a per-block **confidence threshold** (``min_confidence``) drops noisy
    recognitions before they become searchable chunks, and
  - a structured **metrics log line** (``pdf_ocr_metrics``) reports how many
    pages were OCR'd, characters recovered, and the mean confidence — so the
    behavior is diagnosable in production without reverse-engineering.
"""

import json
import logging
from typing import Callable, List, Optional, Sequence, Tuple, Union

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
# Default minimum per-block confidence. 0.0 = keep everything (no behavior change
# vs. an unfiltered engine); operators raise this (e.g. 0.5) to drop OCR noise.
DEFAULT_OCR_MIN_CONFIDENCE = 0.0

# An OCR callable maps a rendered page image (H x W x 3 uint8 array) to either
# plain text or a ``(text, mean_confidence)`` tuple. The tuple form lets the
# default engine surface confidence for metrics; injected fakes may return a
# bare string and confidence is simply reported as unknown.
OcrOutput = Union[str, Tuple[str, Optional[float]]]
OcrCallable = Callable[[np.ndarray], OcrOutput]


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


def _non_ws_chars(text: str) -> int:
    """Count non-whitespace characters in ``text``."""
    return sum(1 for char in text if not char.isspace())


def _parse_ocr_result(
    result, min_confidence: float = DEFAULT_OCR_MIN_CONFIDENCE
) -> Tuple[str, Optional[float]]:
    """Turn a rapidocr result into ``(text, mean_confidence)``.

    rapidocr returns either ``None`` (nothing found) or a list of rows shaped
    ``(box, text, confidence)``. Rows whose confidence is below
    ``min_confidence`` are dropped so low-quality recognitions never reach the
    vector store. Malformed/short rows are tolerated rather than letting an
    ``IndexError`` escape and be reported as a generic failure.

    :returns: the joined kept text, and the mean confidence of the kept rows
        (``None`` when no row carried a usable confidence value).
    """
    if not result:
        return "", None
    texts: List[str] = []
    confidences: List[float] = []
    for row in result:
        if not (isinstance(row, (list, tuple)) and len(row) >= 2 and row[1]):
            continue
        confidence = None
        if len(row) >= 3:
            try:
                confidence = float(row[2])
            except (TypeError, ValueError):
                confidence = None
        if confidence is not None and confidence < min_confidence:
            continue
        texts.append(str(row[1]))
        if confidence is not None:
            confidences.append(confidence)
    mean_conf = sum(confidences) / len(confidences) if confidences else None
    return "\n".join(texts), mean_conf


def _ocr_result_to_text(
    result, min_confidence: float = DEFAULT_OCR_MIN_CONFIDENCE
) -> str:
    """Defensively turn a rapidocr result into text (confidence-filtered)."""
    return _parse_ocr_result(result, min_confidence)[0]


def _normalize_ocr_output(output: OcrOutput) -> Tuple[str, Optional[float]]:
    """Normalize an OCR callable's return into ``(text, mean_confidence)``.

    Accepts either a bare string (confidence unknown) or a
    ``(text, confidence)`` tuple, so injected test fakes stay simple while the
    default engine can report confidence.
    """
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[0], str):
        text, confidence = output
        return text or "", confidence
    return (output or ""), None


def _build_default_ocr(
    min_confidence: float = DEFAULT_OCR_MIN_CONFIDENCE,
) -> OcrCallable:
    """Construct the default rapidocr-backed OCR callable.

    Imported lazily so that importing this module (and the fast unit tests that
    inject a fake OCR callable) does not pull in the ONNX runtime.
    """
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()

    def run(image: np.ndarray) -> Tuple[str, Optional[float]]:
        result, _elapsed = engine(image)
        return _parse_ocr_result(result, min_confidence)

    return run


def _render_page(page, dpi: int) -> np.ndarray:
    """Rasterize a single pypdfium2 page to an RGB numpy array."""
    bitmap = page.render(scale=dpi / 72.0)
    pil_image = bitmap.to_pil().convert("RGB")
    return np.asarray(pil_image)


def _select_page_indices(
    total_pages: int, max_pages: int, pages: Optional[Sequence[int]], filepath: str
) -> List[int]:
    """Resolve which 0-based page indices to OCR, honoring ``max_pages``.

    When ``pages`` is None, OCR the leading pages up to ``max_pages`` (whole-doc
    fallback). When ``pages`` is given (page-aware fallback for mixed PDFs), OCR
    just those in-range indices, still capped at ``max_pages``.
    """
    if pages is None:
        candidate = list(range(total_pages))
    else:
        candidate = [i for i in pages if 0 <= i < total_pages]

    selected = candidate[:max_pages]
    if len(candidate) > max_pages:
        logger.warning(
            "OCR truncated '%s': %d page(s) to OCR, limit is %d "
            "(raise RAG_PDF_OCR_MAX_PAGES to process them all).",
            filepath,
            len(candidate),
            max_pages,
        )
    return selected


def ocr_pdf(
    filepath: str,
    *,
    dpi: int = DEFAULT_OCR_DPI,
    max_pages: int = DEFAULT_OCR_MAX_PAGES,
    pages: Optional[Sequence[int]] = None,
    min_confidence: float = DEFAULT_OCR_MIN_CONFIDENCE,
    ocr: Optional[OcrCallable] = None,
) -> List[Document]:
    """OCR pages of ``filepath`` and return one Document per OCR'd page.

    :param pages: optional explicit 0-based page indices to OCR. When omitted,
        the leading pages (up to ``max_pages``) are OCR'd. Passing a subset
        enables page-aware fallback for mixed PDFs (some pages have text, some
        are scanned), so only the text-less pages are rasterized.
    :param min_confidence: drop OCR blocks below this confidence (default 0.0,
        i.e. keep everything). Only applied by the default rapidocr engine.
    :param ocr: optional injected OCR callable (defaults to rapidocr).
    :returns: a list of Documents with ``ocr=True`` and a 0-based ``page`` in
        their metadata, returned in the same order as the OCR'd indices. The
        text may be empty for a page if OCR found nothing.
    """
    if ocr is None:
        ocr = _build_default_ocr(min_confidence)

    pdf = pdfium.PdfDocument(filepath)
    try:
        total_pages = len(pdf)
        indices = _select_page_indices(total_pages, max_pages, pages, filepath)

        documents: List[Document] = []
        confidences: List[float] = []
        for index in indices:
            page = pdf[index]
            try:
                image = _render_page(page, dpi)
                text, confidence = _normalize_ocr_output(ocr(image))
            finally:
                page.close()
            if confidence is not None:
                confidences.append(confidence)
            documents.append(
                Document(
                    page_content=text,
                    metadata={"source": str(filepath), "page": index, "ocr": True},
                )
            )

        _log_metrics(filepath, total_pages, documents, confidences)
        return documents
    finally:
        pdf.close()


def _log_metrics(
    filepath: str,
    total_pages: int,
    documents: Sequence[Document],
    confidences: Sequence[float],
) -> None:
    """Emit a structured, machine-parseable summary of an OCR pass.

    A single JSON line makes ingestion behavior diagnosable in production
    (how many pages were OCR'd, how much text was recovered, how confident the
    engine was) without having to reverse-engineer the loaders.
    """
    chars_extracted = sum(_non_ws_chars(doc.page_content) for doc in documents)
    mean_conf = (
        round(sum(confidences) / len(confidences), 3) if confidences else None
    )
    metrics = {
        "event": "pdf_ocr_metrics",
        "source": str(filepath).split("/")[-1].split("\\")[-1],
        "pages_total": total_pages,
        "pages_ocr_applied": len(documents),
        "chars_extracted": chars_extracted,
        "ocr_confidence_avg": mean_conf,
    }
    logger.info("pdf_ocr_metrics %s", json.dumps(metrics))
