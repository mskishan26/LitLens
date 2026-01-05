import yaml
import os
import re
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively resolve ${section.key} variables."""
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
            matches = re.findall(r'\${([^}]+)}', item)
            for match in matches:
                val = _get_value(match, root_config)
                if val is not None:
                    item = item.replace(f'${{{match}}}', str(val))
            if item.startswith('~/'):
                item = str(Path.home() / item[2:])
            return item
        return item

    return _replace_vars(config, config)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    # 1. Standard search for config.yaml
    if config_path is None:
        possible_paths = [Path("config.yaml"), Path("/root/app/config.yaml"), Path(__file__).parent / "config.yaml"]
        config_path = next((p for p in possible_paths if p.exists()), Path("config.yaml"))

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Environment Detection
    is_modal = os.environ.get("MODAL_IMAGE_ID") is not None
    
    if is_modal:
        # Cloud paths
        config['paths']['data_root'] = config['paths']['modal_data_path']
        model_base = config['paths']['modal_model_path']
    else:
        # Local/Explorer paths
        # Note: data_root is already in your YAML as "/mnt/e/data_files"
        model_base = config['paths']['local_model_path']
    os.environ['HF_HOME'] = model_base
    model_base = Path(model_base)

    # 3. Dynamic Model Path Construction
    # Converts 'org/name' to 'models--org--name' structure
    if 'models' in config:
        for m_type, m_info in config['models'].items():
            if isinstance(m_info, dict) and 'id' in m_info and 'revision' in m_info:
                hf_folder = f"models--{m_info['id'].replace('/', '--')}"
                full_path = model_base / hf_folder / "snapshots" / m_info['revision']
                config['models'][m_type]['path'] = str(full_path)

    # 4. Final variable resolution (like ${paths.data_root})
    config = _resolve_paths(config)
    
    return config