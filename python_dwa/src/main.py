#!/usr/bin/env python3
"""
Dynamic Window Approach (DWA) Simulation

Main entry point for DWA local planning simulation with differential drive robot.

This simulation demonstrates:
- Dynamic Window Approach for local path planning
- Differential drive kinematics
- Dynamic obstacle avoidance
- Real-time trajectory visualization
- Optional GIF recording

Usage:
    python main.py [--config path/to/config.yaml]

Authors: Baraa Lazkani, Modar Ibrahim, Laith Alsheikh
"""

import argparse
import time
import sys

from robot import DifferentialDriveRobot
from environment import Environment
from dwa_planner import DWAPlanner
from visualizer import Visualizer
from config_loader import (load_config, get_robot_params, get_planning_params,
                           get_environment_params, get_visualization_params)


def main():
    """Main simulation loop."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Dynamic Window Approach Simulation')
    parser.add_argument('--config', type=str, default='../configs/dwa.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Extract parameters
    robot_params = get_robot_params(config)
    planning_params = get_planning_params(config)
    env_params = get_environment_params(config)
    vis_params = get_visualization_params(config)

    print("\n=== Dynamic Window Approach Simulation ===")
    print(f"Robot radius: {robot_params['radius']} m")
    print(f"Max velocity: {robot_params['max_velocity']} m/s")
    print(f"Planning horizon: {planning_params['timestep'] * planning_params['steps_ahead']} s")
    print(f"Number of dynamic obstacles: {env_params['num_dynamic']}")
    print(f"Recording GIF: {vis_params['record_gif']}")
    print("==========================================\n")

    # Initialize robot
    robot = DifferentialDriveRobot(
        radius=robot_params['radius'],
        wheel_width=robot_params['wheel_width'],
        max_velocity=robot_params['max_velocity'],
        max_acceleration=robot_params['max_acceleration'],
        safe_distance=robot_params['safe_distance']
    )

    # Set initial robot pose (start on left edge)
    initial_x = env_params['playfield'][0] - 0.5
    initial_y = 0.0
    initial_theta = 0.0
    robot.set_pose(initial_x, initial_y, initial_theta)

    # Initialize environment
    environment = Environment(
        playfield_bounds=env_params['playfield'],
        barrier_radius=env_params['barrier_radius'],
        robot_radius=robot_params['radius']
    )

    # Add obstacles
    environment.add_static_obstacles(env_params['static_obstacles'])
    environment.generate_dynamic_obstacles(
        num_obstacles=env_params['num_dynamic'],
        velocity_range=env_params['velocity_range']
    )

    print(f"Generated {len(environment.barriers)} dynamic obstacles")
    print(f"Target obstacle: {environment.target_index}")

    # Initialize planner
    planner = DWAPlanner(
        robot=robot,
        environment=environment,
        timestep=planning_params['timestep'],
        steps_ahead=planning_params['steps_ahead'],
        forward_weight=planning_params['forward_weight'],
        obstacle_weight=planning_params['obstacle_weight']
    )

    # Initialize visualizer
    visualizer = Visualizer(
        width=vis_params['width'],
        height=vis_params['height'],
        scale=vis_params['scale'],
        record_gif=vis_params['record_gif'],
        gif_duration=vis_params['gif_duration'],
        frame_skip=vis_params['frame_skip']
    )

    # Start recording if enabled
    visualizer.start_recording()

    print("\nSimulation running... (Close window to exit)")

    # Main simulation loop
    dt = planning_params['timestep']
    try:
        while True:
            # Handle events
            if not visualizer.handle_events():
                break

            # Plan next action using DWA
            vL, vR = planner.plan()

            # Update robot state
            robot.update(vL, vR, dt)

            # Update environment (move obstacles)
            environment.update_obstacles(dt)

            # Check if target reached
            x, y, _ = robot.get_pose()
            if environment.check_target_reached(x, y, robot.radius):
                print(f"Target {environment.target_index} reached! Setting new target...")
                environment.set_random_target()
                robot.location_history = []  # Clear trail

            # Render scene
            paths, positions = planner.get_visualization_data()
            visualizer.render(robot, environment, paths, positions)

            # Control simulation speed
            time.sleep(dt / 50)

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")

    finally:
        # Cleanup
        visualizer.cleanup()
        print("Simulation ended.")


if __name__ == "__main__":
    main()
