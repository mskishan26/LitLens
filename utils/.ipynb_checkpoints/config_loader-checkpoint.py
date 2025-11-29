import yaml
import os
from pathlib import Path
from typing import Any, Dict
import re
import sys

def _resolve_paths(config: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
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
            
            # Handle ~ expansion
            if item.startswith('~/'):
                item = str(Path.home() / item[2:])
            
            return item
        else:
            return item

    return _replace_vars(config, config)

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from yaml file.
    
    Args:
        config_path: Path to config file. If None, looks for config.yaml in project root.
        
    Returns:
        Dictionary containing configuration.
    """
    if config_path is None:
        # Assume config.yaml is in the src directory (parent of this file's directory)
        current_dir = Path(__file__).parent
        project_root = current_dir.parent
        config_path = project_root / 'config.yaml'
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Resolve variable substitutions
    config = _resolve_paths(config, config_path.parent)
    
    # Set HF_HOME if configured
    if 'paths' in config and 'huggingface_home' in config['paths']:
        os.environ['HF_HOME'] = config['paths']['huggingface_home']

    if 'paths' in config and 'logs' in config['paths']:
        os.environ['RAG_LOG_DIR'] = config['paths']['logs']
        
    # Add workdir to sys.path if configured
    if 'paths' in config and 'workdir' in config['paths']:
        workdir = Path(config['paths']['workdir']).resolve()
        if workdir.exists():
            workdir_str = str(workdir)
            if workdir_str not in sys.path:
                sys.path.append(workdir_str)
            
            # Export PYTHONPATH for subprocesses
            current_pythonpath = os.environ.get('PYTHONPATH', '')
            if workdir_str not in current_pythonpath.split(os.pathsep):
                os.environ['PYTHONPATH'] = f"{workdir_str}{os.pathsep}{current_pythonpath}"
    
    return config
