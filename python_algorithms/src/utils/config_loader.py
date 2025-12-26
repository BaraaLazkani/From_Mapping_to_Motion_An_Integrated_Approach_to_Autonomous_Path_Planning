"""
YAML configuration file loader for path planning algorithms.

This module provides utilities to load and validate YAML configuration files
for environment setup and algorithm parameters.
"""

import yaml
from typing import Dict, Any
from pathlib import Path


def load_yaml_config(filepath: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        filepath: Path to YAML file

    Returns:
        Dictionary containing configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If file is not valid YAML

    Example:
        >>> config = load_yaml_config('configs/environment.yaml')
        >>> print(config['environment']['map_size'])
        {'width': 20.0, 'height': 20.0}
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with open(filepath, 'r') as f:
        try:
            config = yaml.safe_load(f)
            return config if config is not None else {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {filepath}: {e}")


def load_environment_config(config_dir: str = 'configs') -> Dict[str, Any]:
    """
    Load environment configuration from YAML file.

    Args:
        config_dir: Directory containing config files (default: 'configs')

    Returns:
        Dictionary with environment parameters:
        - map_size: {width, height}
        - start_point: {x, y}
        - goal_point: {x, y}
        - obstacles: List of {center, size}

    Example:
        >>> env_config = load_environment_config()
        >>> map_size = (env_config['map_size']['width'],
        ...             env_config['map_size']['height'])
    """
    config_path = Path(config_dir) / 'environment.yaml'
    config = load_yaml_config(str(config_path))
    return config.get('environment', {})


def load_algorithm_config(algorithm_name: str, config_dir: str = 'configs') -> Dict[str, Any]:
    """
    Load algorithm-specific configuration from YAML file.

    Args:
        algorithm_name: Name of algorithm ('astar', 'dijkstra', 'rrt_star', 'apf')
        config_dir: Directory containing config files (default: 'configs')

    Returns:
        Dictionary with algorithm-specific parameters

    Raises:
        FileNotFoundError: If algorithm config file doesn't exist

    Example:
        >>> astar_config = load_algorithm_config('astar')
        >>> heuristic = astar_config['parameters']['heuristic_type']
    """
    config_path = Path(config_dir) / f'{algorithm_name}.yaml'
    config = load_yaml_config(str(config_path))
    return config.get('algorithm', {})


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.

    Later dictionaries override earlier ones for conflicting keys.

    Args:
        *configs: Variable number of configuration dictionaries

    Returns:
        Merged configuration dictionary

    Example:
        >>> base_config = {'a': 1, 'b': 2}
        >>> override_config = {'b': 3, 'c': 4}
        >>> merged = merge_configs(base_config, override_config)
        >>> print(merged)
        {'a': 1, 'b': 3, 'c': 4}
    """
    merged = {}
    for config in configs:
        merged.update(config)
    return merged
