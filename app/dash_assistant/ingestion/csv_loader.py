# app/dash_assistant/ingestion/csv_loader.py
"""CSV loader for dashboards and charts data."""
import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from app.config import logger
from .validators import validate_and_normalize_dashboard, validate_and_normalize_chart, ValidationError


async def load_dashboards_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load dashboards from CSV file.
    
    Args:
        csv_path: Path to dashboards CSV file
        
    Returns:
        List[Dict[str, Any]]: List of dashboard records
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValidationError: If data validation fails
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dashboards CSV file not found: {csv_path}")
    
    dashboards = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 for header
                try:
                    # Clean empty strings to None
                    cleaned_row = {k: v.strip() if v.strip() else None for k, v in row.items()}
                    
                    # Validate and normalize
                    normalized_dashboard = validate_and_normalize_dashboard(cleaned_row)
                    dashboards.append(normalized_dashboard)
                    
                except ValidationError as e:
                    logger.error(f"Validation error in dashboards CSV row {row_num}: {e.errors}")
                    raise
                except Exception as e:
                    logger.error(f"Error processing dashboards CSV row {row_num}: {e}")
                    raise
    
    except Exception as e:
        logger.error(f"Error reading dashboards CSV file {csv_path}: {e}")
        raise
    
    logger.info(f"Loaded {len(dashboards)} dashboards from {csv_path}")
    return dashboards


async def load_charts_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load charts from CSV file.
    
    Args:
        csv_path: Path to charts CSV file
        
    Returns:
        List[Dict[str, Any]]: List of chart records
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValidationError: If data validation fails
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Charts CSV file not found: {csv_path}")
    
    charts = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row_num, row in enumerate(reader, start=2):  # Start at 2 for header
                try:
                    # Clean empty strings to None
                    cleaned_row = {k: v.strip() if v.strip() else None for k, v in row.items()}
                    
                    # Parse JSON fields if they exist
                    json_fields = ['metrics', 'dimensions', 'filters_default']
                    for field in json_fields:
                        if cleaned_row.get(field):
                            try:
                                cleaned_row[field] = json.loads(cleaned_row[field])
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON in {field} for chart row {row_num}: {e}")
                                cleaned_row[field] = None
                    
                    # Validate and normalize
                    normalized_chart = validate_and_normalize_chart(cleaned_row)
                    charts.append(normalized_chart)
                    
                except ValidationError as e:
                    logger.error(f"Validation error in charts CSV row {row_num}: {e.errors}")
                    raise
                except Exception as e:
                    logger.error(f"Error processing charts CSV row {row_num}: {e}")
                    raise
    
    except Exception as e:
        logger.error(f"Error reading charts CSV file {csv_path}: {e}")
        raise
    
    logger.info(f"Loaded {len(charts)} charts from {csv_path}")
    return charts


def parse_csv_tags(tags_str: str) -> List[str]:
    """Parse tags from CSV string format.
    
    Args:
        tags_str: Tags string (comma-separated or JSON array)
        
    Returns:
        List[str]: List of tags
    """
    if not tags_str:
        return []
    
    tags_str = tags_str.strip()
    
    # Try to parse as JSON array first
    if tags_str.startswith('[') and tags_str.endswith(']'):
        try:
            return json.loads(tags_str)
        except json.JSONDecodeError:
            pass
    
    # Fall back to comma-separated parsing
    return [tag.strip() for tag in tags_str.split(',') if tag.strip()]


def validate_csv_structure(csv_path: Path, required_columns: List[str]) -> bool:
    """Validate CSV file structure.
    
    Args:
        csv_path: Path to CSV file
        required_columns: List of required column names
        
    Returns:
        bool: True if structure is valid
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If required columns are missing
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            # Check if all required columns are present
            missing_columns = set(required_columns) - set(reader.fieldnames or [])
            if missing_columns:
                raise ValueError(f"Missing required columns in {csv_path}: {missing_columns}")
            
            return True
    
    except Exception as e:
        logger.error(f"Error validating CSV structure for {csv_path}: {e}")
        raise


async def validate_dashboards_csv(csv_path: Path) -> bool:
    """Validate dashboards CSV file structure.
    
    Args:
        csv_path: Path to dashboards CSV file
        
    Returns:
        bool: True if valid
    """
    required_columns = ['superset_id', 'title']
    return validate_csv_structure(csv_path, required_columns)


async def validate_charts_csv(csv_path: Path) -> bool:
    """Validate charts CSV file structure.
    
    Args:
        csv_path: Path to charts CSV file
        
    Returns:
        bool: True if valid
    """
    required_columns = ['superset_chart_id', 'parent_dashboard_id', 'title']
    return validate_csv_structure(csv_path, required_columns)


class CSVLoadError(Exception):
    """Custom exception for CSV loading errors."""
    pass


async def load_and_validate_dashboards_csv(csv_path: Path, 
                                         domain_mapping: Optional[Dict[str, str]] = None,
                                         default_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load and validate dashboards CSV with custom settings.
    
    Args:
        csv_path: Path to dashboards CSV file
        domain_mapping: Optional domain mapping rules
        default_tags: Optional default tags to add
        
    Returns:
        List[Dict[str, Any]]: List of validated dashboard records
    """
    # Validate structure first
    await validate_dashboards_csv(csv_path)
    
    # Load data
    dashboards = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row_num, row in enumerate(reader, start=2):
                try:
                    # Clean empty strings to None
                    cleaned_row = {k: v.strip() if v.strip() else None for k, v in row.items()}
                    
                    # Parse tags if present
                    if cleaned_row.get('tags'):
                        cleaned_row['tags'] = parse_csv_tags(cleaned_row['tags'])
                    
                    # Validate and normalize with custom settings
                    normalized_dashboard = validate_and_normalize_dashboard(
                        cleaned_row, 
                        domain_mapping=domain_mapping,
                        default_tags=default_tags
                    )
                    dashboards.append(normalized_dashboard)
                    
                except ValidationError as e:
                    error_msg = f"Validation error in dashboards CSV row {row_num}: {e.errors}"
                    logger.error(error_msg)
                    raise CSVLoadError(error_msg)
                except Exception as e:
                    error_msg = f"Error processing dashboards CSV row {row_num}: {e}"
                    logger.error(error_msg)
                    raise CSVLoadError(error_msg)
    
    except CSVLoadError:
        raise
    except Exception as e:
        error_msg = f"Error reading dashboards CSV file {csv_path}: {e}"
        logger.error(error_msg)
        raise CSVLoadError(error_msg)
    
    logger.info(f"Successfully loaded and validated {len(dashboards)} dashboards from {csv_path}")
    return dashboards


async def load_and_validate_charts_csv(csv_path: Path) -> List[Dict[str, Any]]:
    """Load and validate charts CSV.
    
    Args:
        csv_path: Path to charts CSV file
        
    Returns:
        List[Dict[str, Any]]: List of validated chart records
    """
    # Validate structure first
    await validate_charts_csv(csv_path)
    
    # Load data
    charts = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row_num, row in enumerate(reader, start=2):
                try:
                    # Clean empty strings to None
                    cleaned_row = {k: v.strip() if v.strip() else None for k, v in row.items()}
                    
                    # Parse JSON fields if they exist
                    json_fields = ['metrics', 'dimensions', 'filters_default']
                    for field in json_fields:
                        if cleaned_row.get(field):
                            try:
                                cleaned_row[field] = json.loads(cleaned_row[field])
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON in {field} for chart row {row_num}: {e}")
                                cleaned_row[field] = None
                    
                    # Validate and normalize
                    normalized_chart = validate_and_normalize_chart(cleaned_row)
                    charts.append(normalized_chart)
                    
                except ValidationError as e:
                    error_msg = f"Validation error in charts CSV row {row_num}: {e.errors}"
                    logger.error(error_msg)
                    raise CSVLoadError(error_msg)
                except Exception as e:
                    error_msg = f"Error processing charts CSV row {row_num}: {e}"
                    logger.error(error_msg)
                    raise CSVLoadError(error_msg)
    
    except CSVLoadError:
        raise
    except Exception as e:
        error_msg = f"Error reading charts CSV file {csv_path}: {e}"
        logger.error(error_msg)
        raise CSVLoadError(error_msg)
    
    logger.info(f"Successfully loaded and validated {len(charts)} charts from {csv_path}")
    return charts
