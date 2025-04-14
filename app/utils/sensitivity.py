import os
from typing import Optional
from fastapi import HTTPException
from dotenv import load_dotenv
from app.config import logger
import zipfile
import pikepdf
from lxml import etree
from xml.etree import ElementTree as ET

# Load .env
load_dotenv()

def get_env_list(key: str) -> list[str]:
    raw_value = os.getenv(key)
    if raw_value:
        return [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    return []  # Return an empty list if the key is not found

# Configuration for allowed document types and labels
ALLOWED_LABELS = get_env_list("ALLOWED_LABELS")
CHECKED_DOC_TYPES = get_env_list("CHECKED_DOC_TYPES")

# Define the supported document types for checking (by default, all these are checked if no CHECKED_DOC_TYPES is defined)
SUPPORTED_DOC_TYPES = ["pdf", "docx", "xlsx", "pptx"]

def normalize_label(label: Optional[str]) -> str:
    return label.strip().lower() if label else ""

def is_label_allowed(label: Optional[str]) -> bool:
    # If no allowed labels are defined, allow all labels
    if not ALLOWED_LABELS:
        return True

    if label is None:
        return "none" in ALLOWED_LABELS

    normalized = normalize_label(label)
    return any(allowed in normalized for allowed in ALLOWED_LABELS)

def is_doc_type_allowed(filename: str) -> bool:
    """
    Check if the document type (based on its file extension) is allowed according to the config.
    """
    file_ext = filename.split('.')[-1].lower()
    
    # If CHECKED_DOC_TYPES is defined, check if the file type is in the allowed list
    if CHECKED_DOC_TYPES:
        return file_ext in CHECKED_DOC_TYPES
    
    # If CHECKED_DOC_TYPES is not defined, allow all document types by default
    return file_ext in SUPPORTED_DOC_TYPES

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
    # First, check if the file type is allowed
    if not is_doc_type_allowed(filename):
        logger.warning(f"Document type {filename.split('.')[-1]} is not allowed for sensitivity check.")
        return None

    # Proceed with the sensitivity label extraction if the type is allowed
    if filename.endswith(".docx") or filename.endswith(".xlsx") or filename.endswith(".pptx"):
        return extract_office_sensitivity_label(file_path)
    elif filename.endswith(".pdf"):
        return extract_pdf_sensitivity_label(file_path)
    
    return None  # Return None if the file type is not supported

def extract_office_sensitivity_label(file_path: str) -> Optional[str]:
    try:
        with zipfile.ZipFile(file_path, "r") as zipf:
            if "docProps/custom.xml" in zipf.namelist():
                with zipf.open("docProps/custom.xml") as custom_file:
                    xml_content = custom_file.read().decode("utf-8")
                    tree = ET.fromstring(xml_content)

                    # Define namespaces
                    ns = {
                        'cp': 'http://schemas.openxmlformats.org/officeDocument/2006/custom-properties',
                        'vt': 'http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes'
                    }

                    # Loop through all property elements
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

            # Define namespace for pdfx (used in your metadata)
            ns = {
                'pdfx': 'http://ns.adobe.com/pdfx/1.3/',
                'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#'
            }

            # Search all rdf:Description elements under rdf:RDF
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
