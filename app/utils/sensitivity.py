import os
from typing import Optional
from fastapi import HTTPException
from dotenv import load_dotenv
from app.config import logger

# Load .env
load_dotenv()

def get_env_list(key: str) -> list[str]:
    raw_value = os.getenv(key)
    if raw_value:
        return [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    return []  # Return an empty list if the key is not found

ALLOWED_LABELS = get_env_list("ALLOWED_LABELS")

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

def assert_sensitivity_allowed(sensitivity_label: str):
    if is_label_allowed(sensitivity_label):
        return

    raise HTTPException(
        status_code=400,
        detail=f"File not processed due to unauthorized sensitivity level: {sensitivity_label}."
    )
