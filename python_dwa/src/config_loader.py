"""
YAML Configuration Loader

Loads simulation parameters from YAML configuration files.
This allows easy experimentation without modifying code.
"""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = "../configs/dwa.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing all configuration parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    # Handle relative paths
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")

    # Validate required sections
    required_sections = ['robot', 'planning', 'environment', 'visualization']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")

    return config


def get_robot_params(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract robot parameters from configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary of robot parameters
    """
    robot = config['robot']
    return {
        'radius': robot['radius'],
        'wheel_width': robot['wheel_width'],
        'max_velocity': robot['max_velocity'],
        'max_acceleration': robot['max_acceleration'],
        'safe_distance': robot['safe_distance']
    }


def get_planning_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract planning parameters from configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary of planning parameters
    """
    planning = config['planning']
    return {
        'timestep': planning['timestep'],
        'steps_ahead': planning['steps_ahead'],
        'forward_weight': planning['forward_weight'],
        'obstacle_weight': planning['obstacle_weight']
    }


def get_environment_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract environment parameters from configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary of environment parameters
    """
    env = config['environment']
    return {
        'playfield': (
            env['playfield']['x_min'],
            env['playfield']['y_min'],
            env['playfield']['x_max'],
            env['playfield']['y_max']
        ),
        'barrier_radius': env['obstacles']['barrier_radius'],
        'num_dynamic': env['obstacles']['num_dynamic'],
        'velocity_range': env['obstacles']['velocity_range'],
        'static_obstacles': [
            (obs['x'], obs['y'])
            for obs in env['obstacles']['static']
        ]
    }


def get_visualization_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract visualization parameters from configuration.

    Args:
        config: Full configuration dictionary

    Returns:
        Dictionary of visualization parameters
    """
    vis = config['visualization']
    return {
        'width': vis['width'],
        'height': vis['height'],
        'scale': vis['scale'],
        'record_gif': vis['record_gif'],
        'gif_duration': vis['gif_duration'],
        'frame_skip': vis['frame_skip']
    }
