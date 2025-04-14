import os
from typing import Optional
from fastapi import HTTPException
from dotenv import load_dotenv
from app.config import logger

# Load .env
load_dotenv()

# Default values
DEFAULT_ALLOWED_LABELS = [
    "public",
    "personal",
    "none"
]

def get_env_list(key: str, fallback: list[str]) -> list[str]:
    raw_value = os.getenv(key)
    if raw_value:
        return [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    return fallback

ALLOWED_LABELS = get_env_list("ALLOWED_LABELS", DEFAULT_ALLOWED_LABELS)


def normalize_label(label: Optional[str]) -> str:
    return label.strip().lower() if label else ""

def is_label_allowed(label: Optional[str]) -> bool:
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
