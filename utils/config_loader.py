import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
import re
import logging

logger = logging.getLogger(__name__)

def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively resolve paths in the config dictionary.
    Replaces ${section.key} with the value from the config.
    """
    def _get_value(path_str: str, current_config: Dict[str, Any]) -> Any:
        keys = path_str.split('.')
        val = current_config
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return None
        return val

    def _replace_vars(item: Any, root_config: Dict[str, Any]) -> Any:
        if isinstance(item, dict):
            return {k: _replace_vars(v, root_config) for k, v in item.items()}
        elif isinstance(item, list):
            return [_replace_vars(i, root_config) for i in item]
        elif isinstance(item, str):
            # Handle ${var} substitution
            matches = re.findall(r'\${([^}]+)}', item)
            for match in matches:
                val = _get_value(match, root_config)
                if val is not None:
                    item = item.replace(f'${{{match}}}', str(val))
                else:
                    logger.warning(f"Could not resolve variable '${{{match}}}' in config.")
            
            # Handle ~ expansion
            if item.startswith('~/'):
                item = str(Path.home() / item[2:])
            
            return item
        else:
            return item

    return _replace_vars(config, config)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from yaml file.
    Args:
        config_path: Path to config file.
    """
    if config_path is None:
        # Strategy to find config.yaml in various environments (local vs Modal)
        possible_paths = [
            Path("config.yaml"),                    # Current dir
            Path("/root/app/config.yaml"),          # Modal mount root
            Path(__file__).parent.parent / "config.yaml"  # Relative to this file
        ]
        
        for p in possible_paths:
            if p.exists():
                config_path = p
                break
        
        if config_path is None:
            # Last resort
            config_path = Path("config.yaml")

    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Resolve variable substitutions
    config = _resolve_paths(config)
    
    # --- Environment Injection ---
    # Critical for Modal: Set HF_HOME so models are downloaded to the Volume
    if 'paths' in config:
        if 'huggingface_home' in config['paths']:
            hf_home = config['paths']['huggingface_home']
            os.environ['HF_HOME'] = hf_home
            # Ensure it exists
            Path(hf_home).mkdir(parents=True, exist_ok=True)
            
        if 'logs' in config['paths']:
            os.environ['RAG_LOG_DIR'] = config['paths']['logs']
            Path(config['paths']['logs']).mkdir(parents=True, exist_ok=True)
        
    return config