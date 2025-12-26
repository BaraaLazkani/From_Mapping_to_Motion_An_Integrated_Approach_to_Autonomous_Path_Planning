"""
Main entry point for path planning algorithms.

This CLI allows users to run different path planning algorithms with
YAML configuration files.
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.environment import Environment
from core.path_planner import PathPlanner
from algorithms.astar import AStarPlanner
from algorithms.dijkstra import DijkstraPlanner
from algorithms.rrt_star import RRTStarPlanner
from algorithms.apf import APFPlanner
from utils.config_loader import load_environment_config, load_algorithm_config


ALGORITHM_MAP = {
    'astar': AStarPlanner,
    'dijkstra': DijkstraPlanner,
    'rrt_star': RRTStarPlanner,
    'apf': APFPlanner
}


def create_environment_from_config(env_config: dict) -> Environment:
    """
    Create Environment object from configuration dictionary.

    Args:
        env_config: Environment configuration from YAML

    Returns:
        Environment object
    """
    map_size = (env_config['map_size']['width'], env_config['map_size']['height'])

    obstacle_centers = [tuple(obs['center']) for obs in env_config['obstacles']]
    obstacle_sizes = [tuple(obs['size']) for obs in env_config['obstacles']]

    return Environment(map_size, obstacle_centers, obstacle_sizes)


def run_planner(algorithm_name: str, config_dir: str = 'configs', visualize: bool = True, save: bool = False):
    """
    Run a path planning algorithm.

    Args:
        algorithm_name: Name of algorithm ('astar', 'dijkstra', 'rrt_star', 'apf')
        config_dir: Directory containing configuration files
        visualize: Whether to show visualization
        save: Whether to save output files
    """
    if algorithm_name not in ALGORITHM_MAP:
        print(f"Error: Unknown algorithm '{algorithm_name}'")
        print(f"Available algorithms: {', '.join(ALGORITHM_MAP.keys())}")
        return

    print(f"\n{'='*60}")
    print(f"Running {algorithm_name.upper()} Path Planning Algorithm")
    print(f"{'='*60}\n")

    # Load configurations
    print("Loading configurations...")
    env_config = load_environment_config(config_dir)
    alg_config = load_algorithm_config(algorithm_name, config_dir)

    # Create environment
    environment = create_environment_from_config(env_config)
    print(f"Environment: {environment.map_size[0]}x{environment.map_size[1]} with {len(environment.obstacle_centers)} obstacles")

    # Get start and goal
    start = (env_config['start_point']['x'], env_config['start_point']['y'])
    goal = (env_config['goal_point']['x'], env_config['goal_point']['y'])
    print(f"Start: {start}")
    print(f"Goal: {goal}")

    # Create planner
    PlannerClass = ALGORITHM_MAP[algorithm_name]
    planner = PlannerClass(environment, alg_config)
    print(f"Planner: {planner}")

    # Plan path
    print("\nPlanning path...")
    path = planner.plan(start, goal)

    # Display metrics
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    metrics = planner.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")

    if path is None:
        print("❌ No path found!")
        return

    print(f"✅ Path found with {len(path)} waypoints")

    # Visualize
    if visualize:
        fig, ax = plt.subplots(figsize=(10, 6))
        planner.visualize(ax)
        plt.tight_layout()

        if save:
            output_config = alg_config.get('output', {})
            save_path = Path(output_config.get('save_path', f'outputs/{algorithm_name}/'))
            save_path.mkdir(parents=True, exist_ok=True)

            plot_file = save_path / output_config.get('plot_filename', 'path_plot.png')
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            print(f"📊 Plot saved to: {plot_file}")

            # Save path data
            path_file = save_path / 'path.json'
            planner.save_path(str(path_file))
            print(f"💾 Path data saved to: {path_file}")

        plt.show()


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Path Planning Algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run A* algorithm
  python main.py --algorithm astar

  # Run RRT* and save results
  python main.py --algorithm rrt_star --save

  # Run APF without visualization
  python main.py --algorithm apf --no-viz

  # Use custom config directory
  python main.py --algorithm dijkstra --config-dir ../my_configs
        """
    )

    parser.add_argument(
        '--algorithm', '-a',
        type=str,
        choices=list(ALGORITHM_MAP.keys()),
        required=True,
        help='Path planning algorithm to use'
    )

    parser.add_argument(
        '--config-dir', '-c',
        type=str,
        default='configs',
        help='Directory containing YAML configuration files (default: configs)'
    )

    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Save output files (plots, paths, animations)'
    )

    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization'
    )

    args = parser.parse_args()

    # Run the planner
    run_planner(
        algorithm_name=args.algorithm,
        config_dir=args.config_dir,
        visualize=not args.no_viz,
        save=args.save
    )


if __name__ == '__main__':
    main()
