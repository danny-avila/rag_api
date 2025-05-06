import os
from typing import Optional
from fastapi import HTTPException
from dotenv import load_dotenv
from app.config import logger
import zipfile
import pikepdf
from lxml import etree
from xml.etree import ElementTree as ET
import json

# Load .env
load_dotenv()

def get_env_json_list(key: str) -> list[str]:
    raw_value = os.getenv(key)
    try:
        return [item.strip().lower() for item in json.loads(raw_value)] if raw_value else []
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse {key} as JSON list.")
        return []

def get_env_bool(key: str, default: bool = False) -> bool:
    val = os.getenv(key)
    return val.lower() in ("1", "true", "yes") if val is not None else default

# Configuration
DOC_FLTR_ENABLED = get_env_bool("DOC_FLTR_ENABLED")
DOC_FLTR_ALLOWED_LABELS = get_env_json_list("DOC_FLTR_ALLOWED_LABELS")
DOC_FLTR_FILE_TYPES = get_env_json_list("DOC_FLTR_FILE_TYPES")

SUPPORTED_FILE_TYPES = ["pdf", "docx", "xlsx", "pptx"]

def normalize_label(label: Optional[str]) -> str:
    return label.strip().lower() if label else ""

def is_label_allowed(label: Optional[str]) -> bool:
    if label is None:
        return True  # Always allow files with no label

    if not DOC_FLTR_ENABLED:
        return True

    if not DOC_FLTR_ALLOWED_LABELS:
        return True  # If filtering is on but no labels are defined, allow all

    normalized = normalize_label(label)
    return normalized in DOC_FLTR_ALLOWED_LABELS

def is_doc_type_allowed(filename: str) -> bool:
    file_ext = filename.split('.')[-1].lower()
    if DOC_FLTR_FILE_TYPES:
        return file_ext in DOC_FLTR_FILE_TYPES
    return file_ext in SUPPORTED_FILE_TYPES

def assert_sensitivity_allowed(sensitivity_label: str):
    if is_label_allowed(sensitivity_label):
        return

    raise HTTPException(
        status_code=400,
        detail=f"File not processed due to unauthorized sensitivity level: {sensitivity_label}."
    )

# -------------------------------------------------------
# 📁 Sensitivity Label Extractor
# -------------------------------------------------------

async def detect_sensitivity_label(file_path: str, filename: str) -> Optional[str]:
    if not DOC_FLTR_ENABLED:
        return None

    if not is_doc_type_allowed(filename):
        logger.warning(f"Document type {filename.split('.')[-1]} is not allowed for sensitivity check.")
        return None

    if filename.endswith(".docx") or filename.endswith(".xlsx") or filename.endswith(".pptx"):
        return extract_office_sensitivity_label(file_path)
    elif filename.endswith(".pdf"):
        return extract_pdf_sensitivity_label(file_path)

    return None

def extract_office_sensitivity_label(file_path: str) -> Optional[str]:
    try:
        with zipfile.ZipFile(file_path, "r") as zipf:
            if "docProps/custom.xml" in zipf.namelist():
                with zipf.open("docProps/custom.xml") as custom_file:
                    xml_content = custom_file.read().decode("utf-8")
                    tree = ET.fromstring(xml_content)

                    ns = {
                        'cp': 'http://schemas.openxmlformats.org/officeDocument/2006/custom-properties',
                        'vt': 'http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes'
                    }

                    for prop in tree.findall("cp:property", ns):
                        name = prop.attrib.get("name", "")
                        if name.endswith("_Name") or "ClassificationWatermarkText" in name:
                            value_elem = prop.find("vt:lpwstr", ns)
                            return value_elem.text.strip().lower()
    except Exception as e:
        logger.warning("Failed to extract Office label: %s", str(e))

    return None

def extract_pdf_sensitivity_label(file_path: str) -> Optional[str]:
    try:
        with pikepdf.open(file_path) as pdf:
            xmp = pdf.open_metadata()
            xml_content = str(xmp)

            tree = ET.fromstring(xml_content)

            ns = {
                'pdfx': 'http://ns.adobe.com/pdfx/1.3/',
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
            }

            for description in tree.findall('.//rdf:Description', ns):
                for key, value in description.attrib.items():
                    for elem in description:
                        tag = elem.tag
                        if tag.startswith('{%s}' % ns['pdfx']) and tag.endswith('_Name') and elem.text:
                            label = elem.text.strip()
                            logger.info(f"Found sensitivity label: {label}")
                            return label

    except Exception as e:
        logger.warning("Failed to extract PDF label: %s", str(e))

    return None