"""Tests for the OCR fallback used on text-less (scanned / CAD-export) PDFs.

Fast unit tests inject a fake OCR callable so they need no ONNX model. One
integration test (marked ``integration`` + ``slow``) runs the real rapidocr
engine and is excluded by the default pytest options.
"""

from pathlib import Path

import pytest

# ocr.py imports pypdfium2 at module load; skip cleanly where it isn't installed.
pytest.importorskip("pypdfium2")

from langchain_core.documents import Document

from app.utils.ocr import has_text_layer, ocr_pdf, _ocr_result_to_text

FIXTURE = Path(__file__).parent.parent / "fixtures" / "image_only.pdf"


# --- _ocr_result_to_text (defensive against rapidocr return shapes) -------

def test_ocr_result_to_text_none_is_empty():
    assert _ocr_result_to_text(None) == ""


def test_ocr_result_to_text_empty_list_is_empty():
    assert _ocr_result_to_text([]) == ""


def test_ocr_result_to_text_joins_well_formed_rows():
    rows = [(["box"], "line one", 0.99), (["box"], "line two", 0.97)]
    assert _ocr_result_to_text(rows) == "line one\nline two"


def test_ocr_result_to_text_tolerates_malformed_rows():
    rows = [
        (["box"], "good", 0.9),
        (["box"],),          # too short -> skipped, must not IndexError
        (["box"], "", 0.5),  # empty text -> skipped
        (["box"], "also good", 0.8),
    ]
    assert _ocr_result_to_text(rows) == "good\nalso good"


# --- has_text_layer -------------------------------------------------------

def test_has_text_layer_true_for_real_text():
    pages = [Document(page_content="A clear paragraph of real extractable text.")]
    assert has_text_layer(pages) is True


def test_has_text_layer_false_for_empty_pages():
    pages = [Document(page_content=""), Document(page_content="   \n\t  ")]
    assert has_text_layer(pages) is False


def test_has_text_layer_respects_threshold():
    # 5 non-whitespace chars, threshold 6 -> False; threshold 5 -> True
    pages = [Document(page_content="a b c d e")]  # "abcde" == 5 non-ws chars
    assert has_text_layer(pages, min_chars=6) is False
    assert has_text_layer(pages, min_chars=5) is True


def test_has_text_layer_empty_list_is_false():
    assert has_text_layer([]) is False


# --- ocr_pdf --------------------------------------------------------------

def test_ocr_pdf_wires_pages_metadata_with_injected_ocr():
    calls = []

    def fake_ocr(image):
        calls.append(getattr(image, "shape", None))
        return "RECOVERED TEXT"

    docs = ocr_pdf(str(FIXTURE), ocr=fake_ocr)

    assert len(docs) == 1  # fixture is single page
    assert docs[0].page_content == "RECOVERED TEXT"
    assert docs[0].metadata["page"] == 0
    assert docs[0].metadata["ocr"] is True
    assert docs[0].metadata["source"] == str(FIXTURE)
    # the fake received an actual rendered image array
    assert calls and calls[0] is not None


def test_ocr_pdf_respects_max_pages():
    docs = ocr_pdf(str(FIXTURE), max_pages=0, ocr=lambda image: "x")
    assert docs == []


def test_ocr_pdf_blank_ocr_yields_blank_pages():
    docs = ocr_pdf(str(FIXTURE), ocr=lambda image: "")
    assert len(docs) == 1
    assert docs[0].page_content == ""
    # the recovered text is empty -> caller (loader) decides this is a failure
    assert has_text_layer(docs) is False


# --- integration: real OCR engine -----------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_ocr_pdf_recovers_text_with_real_engine():
    docs = ocr_pdf(str(FIXTURE))
    recovered = " ".join(d.page_content for d in docs).lower()
    # tokens baked into tests/fixtures/image_only.pdf
    for token in ("bolt", "m8", "torque", "12", "valve"):
        assert token in recovered, f"missing {token!r} in OCR output: {recovered!r}"
    assert has_text_layer(docs) is True
