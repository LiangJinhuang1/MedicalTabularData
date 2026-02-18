import yaml
from pathlib import Path
from typing import Tuple, Union, Dict, Any

# Get the project root directory
PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent

def load_config(config_path):
    """
    Load YAML configuration file.
    """
    config_path = Path(config_path)
    
    # If relative path, resolve it relative to project root
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    
    # Ensure the path exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_config_value(config, *keys, default=None):
    """
    Helper to safely get nested config values.
    """
    if not isinstance(config, dict):
        return default
    
    if not keys:
        return config if config is not None else default
    
    # Traverse the key path
    current = config
    for i, key in enumerate(keys):
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]
    
    # Return the final value (or default if None)
    return current if current is not None else default


def resolve_path(file_path, base_dir=None):
    """
    Resolve file path flexibly.
    """
    if not file_path:
        return None
    
    file_path = Path(file_path)
    
    # If absolute path, return as is
    if file_path.is_absolute():
        return file_path
    
    # If relative path, resolve relative to base_dir (default: PROJECT_ROOT)
    base_dir = PROJECT_ROOT if base_dir is None else Path(base_dir).absolute()
    return base_dir / file_path


def get_variable_types(data_config):
    """
    Convenience helper to extract (continuous, binary, categorical) lists from data_config.
    Returns a tuple: (continuous_cols, binary_cols, categorical_cols)
    """
    var_types = get_config_value(data_config, 'data', 'variable_types', default={})
    return (
        var_types.get('continuous', []),
        var_types.get('binary', []),
        var_types.get('categorical', []),
    )


def resolve_experiment_paths(
    experiment_path: Union[str, Path],
    config_name: str = "full_config.yaml",
) -> Tuple[Path, Path]:
    """
    Resolve experiment directory and config path from a file or directory input.
    Returns (experiment_dir, config_path).
    """
    experiment_path = Path(experiment_path)
    if experiment_path.is_file():
        config_path = experiment_path
        experiment_dir = config_path.parent.parent if config_path.parent.name == 'configs' else config_path.parent
    else:
        experiment_dir = experiment_path
        config_path = experiment_dir / 'configs' / config_name
    return experiment_dir, config_path


def load_experiment_config(
    experiment_path: Union[str, Path],
    config_name: str = "full_config.yaml",
    require_exists: bool = True,
) -> Tuple[Dict[str, Any], Path, Path]:
    """
    Load the experiment config and return (full_config, experiment_dir, config_path).
    """
    experiment_dir, config_path = resolve_experiment_paths(experiment_path, config_name=config_name)
    if require_exists and not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    full_config = load_config(config_path)
    return full_config, experiment_dir, config_path
