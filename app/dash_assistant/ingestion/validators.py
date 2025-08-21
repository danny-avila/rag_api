# app/dash_assistant/ingestion/validators.py
"""Validators and normalizers for ingestion data."""
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse


def normalize_slug(slug: str) -> str:
    """Normalize dashboard slug to consistent format.
    
    Args:
        slug: Raw slug string
        
    Returns:
        str: Normalized slug
    """
    if not slug:
        return ""
    
    # Convert to lowercase and replace spaces/underscores with hyphens
    normalized = re.sub(r'[_\s]+', '-', slug.lower())
    
    # Remove special characters except hyphens and alphanumeric
    normalized = re.sub(r'[^a-z0-9\-]', '', normalized)
    
    # Remove multiple consecutive hyphens
    normalized = re.sub(r'-+', '-', normalized)
    
    # Remove leading/trailing hyphens
    normalized = normalized.strip('-')
    
    return normalized


def normalize_domain(domain: str, domain_mapping: Optional[Dict[str, str]] = None) -> str:
    """Normalize domain name using mapping rules.
    
    Args:
        domain: Raw domain string
        domain_mapping: Optional mapping for domain normalization
        
    Returns:
        str: Normalized domain
    """
    if not domain:
        return ""
    
    normalized = domain.lower().strip()
    
    # Apply domain mapping if provided
    if domain_mapping and normalized in domain_mapping:
        normalized = domain_mapping[normalized]
    
    return normalized


def normalize_tags(tags: List[str], default_tags: Optional[List[str]] = None) -> List[str]:
    """Normalize and deduplicate tags.
    
    Args:
        tags: List of tag strings
        default_tags: Optional default tags to add
        
    Returns:
        List[str]: Normalized and deduplicated tags
    """
    if not tags:
        tags = []
    
    # Add default tags if provided
    if default_tags:
        tags = list(tags) + list(default_tags)
    
    # Normalize each tag
    normalized_tags = []
    for tag in tags:
        if isinstance(tag, str) and tag.strip():
            # Convert to lowercase, remove special characters
            normalized_tag = re.sub(r'[^a-z0-9\-_]', '', tag.lower().strip())
            if normalized_tag:
                normalized_tags.append(normalized_tag)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_tags = []
    for tag in normalized_tags:
        if tag not in seen:
            seen.add(tag)
            unique_tags.append(tag)
    
    return unique_tags


def validate_url(url: str) -> bool:
    """Validate URL format.
    
    Args:
        url: URL string to validate
        
    Returns:
        bool: True if valid URL, False otherwise
    """
    if not url:
        return False
    
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_superset_id(superset_id: str) -> bool:
    """Validate Superset ID format.
    
    Args:
        superset_id: Superset ID to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not superset_id:
        return False
    
    # Should be numeric string
    return superset_id.isdigit()


def validate_dashboard_data(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate dashboard data and return validation errors.
    
    Args:
        data: Dashboard data dictionary
        
    Returns:
        Dict[str, List[str]]: Dictionary of field names to error messages
    """
    errors = {}
    
    # Required fields
    required_fields = ['title', 'superset_id']
    for field in required_fields:
        if not data.get(field):
            errors.setdefault(field, []).append(f"{field} is required")
    
    # Validate superset_id format
    if data.get('superset_id') and not validate_superset_id(data['superset_id']):
        errors.setdefault('superset_id', []).append("superset_id must be numeric")
    
    # Validate URL if provided
    if data.get('url') and not validate_url(data['url']):
        errors.setdefault('url', []).append("Invalid URL format")
    
    # Validate title length
    if data.get('title') and len(data['title']) > 255:
        errors.setdefault('title', []).append("Title too long (max 255 characters)")
    
    # Validate description length
    if data.get('description') and len(data['description']) > 2000:
        errors.setdefault('description', []).append("Description too long (max 2000 characters)")
    
    return errors


def validate_chart_data(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate chart data and return validation errors.
    
    Args:
        data: Chart data dictionary
        
    Returns:
        Dict[str, List[str]]: Dictionary of field names to error messages
    """
    errors = {}
    
    # Required fields
    required_fields = ['title', 'superset_chart_id', 'parent_dashboard_id']
    for field in required_fields:
        if not data.get(field):
            errors.setdefault(field, []).append(f"{field} is required")
    
    # Validate superset_chart_id format
    if data.get('superset_chart_id') and not validate_superset_id(data['superset_chart_id']):
        errors.setdefault('superset_chart_id', []).append("superset_chart_id must be numeric")
    
    # Validate parent_dashboard_id format
    if data.get('parent_dashboard_id') and not validate_superset_id(data['parent_dashboard_id']):
        errors.setdefault('parent_dashboard_id', []).append("parent_dashboard_id must be numeric")
    
    # Validate URL if provided
    if data.get('url') and not validate_url(data['url']):
        errors.setdefault('url', []).append("Invalid URL format")
    
    # Validate title length
    if data.get('title') and len(data['title']) > 255:
        errors.setdefault('title', []).append("Title too long (max 255 characters)")
    
    return errors


def clean_sql_text(sql: str) -> str:
    """Clean and normalize SQL text.
    
    Args:
        sql: Raw SQL string
        
    Returns:
        str: Cleaned SQL string
    """
    if not sql:
        return ""
    
    # Remove extra whitespace and normalize line endings
    cleaned = re.sub(r'\s+', ' ', sql.strip())
    
    # Remove comments (simple approach)
    cleaned = re.sub(r'--.*$', '', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    return cleaned.strip()


def extract_sql_keywords(sql: str) -> List[str]:
    """Extract SQL keywords and table names for indexing.
    
    Args:
        sql: SQL query string
        
    Returns:
        List[str]: List of extracted keywords
    """
    if not sql:
        return []
    
    # Common SQL keywords to extract
    sql_keywords = [
        'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING',
        'JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'INNER JOIN', 'OUTER JOIN',
        'UNION', 'DISTINCT', 'COUNT', 'SUM', 'AVG', 'MAX', 'MIN',
        'DATE_TRUNC', 'EXTRACT', 'CASE', 'WHEN', 'THEN', 'ELSE'
    ]
    
    keywords_found = []
    sql_upper = sql.upper()
    
    for keyword in sql_keywords:
        if keyword in sql_upper:
            keywords_found.append(keyword.lower())
    
    # Extract table names (simple heuristic)
    # Look for patterns like "FROM table_name" or "JOIN table_name"
    table_patterns = [
        r'FROM\s+(\w+)',
        r'JOIN\s+(\w+)',
        r'UPDATE\s+(\w+)',
        r'INSERT\s+INTO\s+(\w+)'
    ]
    
    for pattern in table_patterns:
        matches = re.findall(pattern, sql_upper)
        for match in matches:
            if match.lower() not in keywords_found:
                keywords_found.append(match.lower())
    
    return keywords_found


class ValidationError(Exception):
    """Custom exception for validation errors."""
    
    def __init__(self, errors: Dict[str, List[str]]):
        self.errors = errors
        super().__init__(f"Validation failed: {errors}")


def validate_and_normalize_dashboard(data: Dict[str, Any], 
                                   domain_mapping: Optional[Dict[str, str]] = None,
                                   default_tags: Optional[List[str]] = None) -> Dict[str, Any]:
    """Validate and normalize dashboard data.
    
    Args:
        data: Raw dashboard data
        domain_mapping: Optional domain mapping rules
        default_tags: Optional default tags to add
        
    Returns:
        Dict[str, Any]: Validated and normalized data
        
    Raises:
        ValidationError: If validation fails
    """
    errors = validate_dashboard_data(data)
    if errors:
        raise ValidationError(errors)
    
    # Normalize data
    normalized = data.copy()
    
    if 'dashboard_slug' in normalized:
        normalized['dashboard_slug'] = normalize_slug(normalized['dashboard_slug'])
    
    if 'domain' in normalized:
        normalized['domain'] = normalize_domain(normalized['domain'], domain_mapping)
    
    if 'tags' in normalized:
        normalized['tags'] = normalize_tags(normalized['tags'], default_tags)
    
    return normalized


def validate_and_normalize_chart(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize chart data.
    
    Args:
        data: Raw chart data
        
    Returns:
        Dict[str, Any]: Validated and normalized data
        
    Raises:
        ValidationError: If validation fails
    """
    errors = validate_chart_data(data)
    if errors:
        raise ValidationError(errors)
    
    # Normalize data
    normalized = data.copy()
    
    if 'sql_text' in normalized:
        normalized['sql_text'] = clean_sql_text(normalized['sql_text'])
    
    return normalized
