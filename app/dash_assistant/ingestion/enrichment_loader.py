# app/dash_assistant/ingestion/enrichment_loader.py
"""YAML enrichment loader for dashboard metadata enhancement."""
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from app.config import logger
from .validators import normalize_domain, normalize_tags


async def load_enrichment_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Load enrichment configuration from YAML file.
    
    Args:
        yaml_path: Path to enrichment YAML file
        
    Returns:
        Dict[str, Any]: Enrichment configuration
        
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Enrichment YAML file not found: {yaml_path}")
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            enrichment_config = yaml.safe_load(file)
        
        # Validate and normalize the configuration
        normalized_config = normalize_enrichment_config(enrichment_config)
        
        logger.info(f"Loaded enrichment configuration from {yaml_path}")
        return normalized_config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {yaml_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading enrichment YAML {yaml_path}: {e}")
        raise


def normalize_enrichment_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize enrichment configuration.
    
    Args:
        config: Raw enrichment configuration
        
    Returns:
        Dict[str, Any]: Normalized configuration
    """
    normalized = {
        'dashboards': {},
        'global_rules': {}
    }
    
    # Process dashboard-specific enrichments
    if 'dashboards' in config:
        for slug, enrichment in config['dashboards'].items():
            normalized_enrichment = normalize_dashboard_enrichment(enrichment)
            normalized['dashboards'][slug] = normalized_enrichment
    
    # Process global rules
    if 'global_rules' in config:
        normalized['global_rules'] = normalize_global_rules(config['global_rules'])
    
    return normalized


def normalize_dashboard_enrichment(enrichment: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize dashboard-specific enrichment data.
    
    Args:
        enrichment: Raw enrichment data for a dashboard
        
    Returns:
        Dict[str, Any]: Normalized enrichment data
    """
    normalized = {}
    
    # Normalize domain
    if 'domain' in enrichment:
        normalized['domain'] = normalize_domain(enrichment['domain'])
    
    # Normalize tags
    if 'tags' in enrichment:
        tags = enrichment['tags']
        if isinstance(tags, str):
            tags = [tags]
        normalized['tags'] = normalize_tags(tags)
    
    # Copy other fields as-is
    for key, value in enrichment.items():
        if key not in ['domain', 'tags']:
            normalized[key] = value
    
    return normalized


def normalize_global_rules(rules: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize global enrichment rules.
    
    Args:
        rules: Raw global rules
        
    Returns:
        Dict[str, Any]: Normalized global rules
    """
    normalized = {}
    
    # Normalize default tags
    if 'default_tags' in rules:
        default_tags = rules['default_tags']
        if isinstance(default_tags, str):
            default_tags = [default_tags]
        normalized['default_tags'] = normalize_tags(default_tags)
    
    # Copy domain mapping as-is (will be used by normalize_domain)
    if 'domain_mapping' in rules:
        normalized['domain_mapping'] = rules['domain_mapping']
    
    # Copy other rules as-is
    for key, value in rules.items():
        if key not in ['default_tags', 'domain_mapping']:
            normalized[key] = value
    
    return normalized


def apply_enrichment_to_dashboard(dashboard_data: Dict[str, Any], 
                                enrichment_config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply enrichment configuration to dashboard data.
    
    Args:
        dashboard_data: Dashboard data to enrich
        enrichment_config: Enrichment configuration
        
    Returns:
        Dict[str, Any]: Enriched dashboard data
    """
    enriched = dashboard_data.copy()
    
    # Get dashboard slug for lookup
    slug = dashboard_data.get('dashboard_slug')
    if not slug:
        logger.debug("No dashboard_slug found, skipping enrichment")
        return enriched
    
    # Get global rules
    global_rules = enrichment_config.get('global_rules', {})
    domain_mapping = global_rules.get('domain_mapping', {})
    default_tags = global_rules.get('default_tags', [])
    
    # Apply dashboard-specific enrichment
    dashboard_enrichments = enrichment_config.get('dashboards', {})
    if slug in dashboard_enrichments:
        dashboard_enrichment = dashboard_enrichments[slug]
        
        # Apply domain
        if 'domain' in dashboard_enrichment:
            enriched['domain'] = normalize_domain(
                dashboard_enrichment['domain'], 
                domain_mapping
            )
        
        # Apply tags (merge with existing and defaults)
        existing_tags = enriched.get('tags') or []
        enrichment_tags = dashboard_enrichment.get('tags', [])
        
        # Combine all tags
        all_tags = list(existing_tags) + list(enrichment_tags) + list(default_tags)
        enriched['tags'] = normalize_tags(all_tags)
        
        # Apply other enrichment fields
        for key, value in dashboard_enrichment.items():
            if key not in ['domain', 'tags']:
                enriched[key] = value
    else:
        # Apply only global rules if no specific enrichment
        if default_tags:
            existing_tags = enriched.get('tags') or []
            all_tags = list(existing_tags) + list(default_tags)
            enriched['tags'] = normalize_tags(all_tags)
        
        # Apply domain mapping if domain exists
        if 'domain' in enriched:
            enriched['domain'] = normalize_domain(enriched['domain'], domain_mapping)
    
    return enriched


def get_enrichment_for_slug(enrichment_config: Dict[str, Any], slug: str) -> Dict[str, Any]:
    """Get enrichment data for a specific dashboard slug.
    
    Args:
        enrichment_config: Enrichment configuration
        slug: Dashboard slug
        
    Returns:
        Dict[str, Any]: Enrichment data for the slug
    """
    dashboard_enrichments = enrichment_config.get('dashboards', {})
    return dashboard_enrichments.get(slug, {})


def validate_enrichment_config(config: Dict[str, Any]) -> List[str]:
    """Validate enrichment configuration and return any errors.
    
    Args:
        config: Enrichment configuration to validate
        
    Returns:
        List[str]: List of validation error messages
    """
    errors = []
    
    # Check top-level structure
    if not isinstance(config, dict):
        errors.append("Configuration must be a dictionary")
        return errors
    
    # Validate dashboards section
    if 'dashboards' in config:
        if not isinstance(config['dashboards'], dict):
            errors.append("'dashboards' must be a dictionary")
        else:
            for slug, enrichment in config['dashboards'].items():
                if not isinstance(enrichment, dict):
                    errors.append(f"Enrichment for '{slug}' must be a dictionary")
                    continue
                
                # Validate tags format
                if 'tags' in enrichment:
                    tags = enrichment['tags']
                    if not isinstance(tags, (list, str)):
                        errors.append(f"Tags for '{slug}' must be a list or string")
                
                # Validate domain format
                if 'domain' in enrichment:
                    domain = enrichment['domain']
                    if not isinstance(domain, str):
                        errors.append(f"Domain for '{slug}' must be a string")
    
    # Validate global_rules section
    if 'global_rules' in config:
        if not isinstance(config['global_rules'], dict):
            errors.append("'global_rules' must be a dictionary")
        else:
            global_rules = config['global_rules']
            
            # Validate default_tags
            if 'default_tags' in global_rules:
                default_tags = global_rules['default_tags']
                if not isinstance(default_tags, (list, str)):
                    errors.append("'default_tags' must be a list or string")
            
            # Validate domain_mapping
            if 'domain_mapping' in global_rules:
                domain_mapping = global_rules['domain_mapping']
                if not isinstance(domain_mapping, dict):
                    errors.append("'domain_mapping' must be a dictionary")
                else:
                    for key, value in domain_mapping.items():
                        if not isinstance(key, str) or not isinstance(value, str):
                            errors.append(f"Domain mapping '{key}' -> '{value}' must be string to string")
    
    return errors


async def load_and_validate_enrichment_yaml(yaml_path: Path) -> Dict[str, Any]:
    """Load and validate enrichment YAML file.
    
    Args:
        yaml_path: Path to enrichment YAML file
        
    Returns:
        Dict[str, Any]: Validated enrichment configuration
        
    Raises:
        EnrichmentLoadError: If loading or validation fails
    """
    try:
        # Load the configuration
        config = await load_enrichment_yaml(yaml_path)
        
        # Validate the configuration
        errors = validate_enrichment_config(config)
        if errors:
            error_msg = f"Enrichment configuration validation failed: {'; '.join(errors)}"
            logger.error(error_msg)
            raise EnrichmentLoadError(error_msg)
        
        return config
        
    except EnrichmentLoadError:
        raise
    except Exception as e:
        error_msg = f"Error loading enrichment configuration from {yaml_path}: {e}"
        logger.error(error_msg)
        raise EnrichmentLoadError(error_msg)


def merge_enrichment_configs(base_config: Dict[str, Any], 
                           override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two enrichment configurations.
    
    Args:
        base_config: Base enrichment configuration
        override_config: Override enrichment configuration
        
    Returns:
        Dict[str, Any]: Merged configuration
    """
    merged = {
        'dashboards': {},
        'global_rules': {}
    }
    
    # Merge dashboards
    base_dashboards = base_config.get('dashboards', {})
    override_dashboards = override_config.get('dashboards', {})
    
    # Start with base dashboards
    merged['dashboards'].update(base_dashboards)
    
    # Override with specific dashboard configs
    for slug, enrichment in override_dashboards.items():
        if slug in merged['dashboards']:
            # Merge dashboard-specific enrichment
            base_enrichment = merged['dashboards'][slug]
            merged_enrichment = base_enrichment.copy()
            
            # Merge tags
            if 'tags' in enrichment:
                base_tags = base_enrichment.get('tags', [])
                override_tags = enrichment['tags']
                if isinstance(override_tags, str):
                    override_tags = [override_tags]
                merged_tags = list(base_tags) + list(override_tags)
                merged_enrichment['tags'] = normalize_tags(merged_tags)
            
            # Override other fields
            for key, value in enrichment.items():
                if key != 'tags':
                    merged_enrichment[key] = value
            
            merged['dashboards'][slug] = merged_enrichment
        else:
            merged['dashboards'][slug] = enrichment
    
    # Merge global rules
    base_global = base_config.get('global_rules', {})
    override_global = override_config.get('global_rules', {})
    
    merged['global_rules'] = base_global.copy()
    merged['global_rules'].update(override_global)
    
    # Special handling for default_tags (merge instead of override)
    if 'default_tags' in base_global and 'default_tags' in override_global:
        base_default_tags = base_global['default_tags']
        override_default_tags = override_global['default_tags']
        
        if isinstance(base_default_tags, str):
            base_default_tags = [base_default_tags]
        if isinstance(override_default_tags, str):
            override_default_tags = [override_default_tags]
        
        merged_default_tags = list(base_default_tags) + list(override_default_tags)
        merged['global_rules']['default_tags'] = normalize_tags(merged_default_tags)
    
    return merged


class EnrichmentLoadError(Exception):
    """Custom exception for enrichment loading errors."""
    pass
