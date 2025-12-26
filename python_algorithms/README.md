# Python Path Planning Algorithms

A modular implementation of multiple path planning algorithms for autonomous robot navigation.

## Algorithms Implemented

1. **A*** - Heuristic-guided optimal pathfinding
2. **Dijkstra** - Shortest path without heuristic
3. **RRT*** - Sampling-based planning with rewiring for optimality
4. **APF** - Artificial Potential Field navigation

## Features

- ✅ **Modular Architecture**: Clean separation of concerns with abstract base class
- ✅ **YAML Configuration**: All parameters configurable via YAML files
- ✅ **Comprehensive Documentation**: Well-documented code with docstrings
- ✅ **Visualization**: Built-in matplotlib visualization for all algorithms
- ✅ **Trajectory Smoothing**: CSC (Cubic Spline Curve) smoothing for graph-based algorithms
- ✅ **Performance Metrics**: Path length, planning time, nodes explored

## Directory Structure

```
python_algorithms/
├── src/
│   ├── core/              # Base classes and core components
│   ├── algorithms/        # Algorithm implementations
│   └── utils/             # Shared utilities
├── configs/               # YAML configuration files
├── outputs/               # Algorithm output files (GIFs, plots)
└── tests/                 # Unit tests
```

## Installation

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Interface

```bash
cd python_algorithms

# Run A* algorithm
python src/main.py --algorithm astar

# Run RRT* and save results
python src/main.py --algorithm rrt_star --save

# Run APF without visualization
python src/main.py --algorithm apf --no-viz

# Run Dijkstra with custom config directory
python src/main.py --algorithm dijkstra --config-dir my_configs
```

### Python API

```python
from src.core.environment import Environment
from src.algorithms.astar import AStarPlanner
from src.utils.config_loader import load_algorithm_config

# Create environment
env = Environment(
    map_size=(20.0, 20.0),
    obstacle_centers=[(5.0, 5.0), (12.0, 12.0)],
    obstacle_sizes=[(2.0, 2.0), (3.0, 3.0)]
)

# Load config and create planner
config = load_algorithm_config('astar')
planner = AStarPlanner(env, config)

# Plan path
path = planner.plan(start=(8.0, 1.0), goal=(18.0, 18.0))

# Get metrics
metrics = planner.get_metrics()
print(f"Path length: {metrics['path_length']:.2f}")
print(f"Planning time: {metrics['planning_time']:.3f}s")

# Visualize
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
planner.visualize(ax)
plt.show()
```

## Configuration

### Environment Configuration (`configs/environment.yaml`)

```yaml
environment:
  map_size: {width: 20.0, height: 20.0}
  start_point: {x: 8.0, y: 1.0}
  goal_point: {x: 18.0, y: 18.0}
  obstacles:
    - {center: [5.0, 5.0], size: [2.0, 2.0]}
```

### Algorithm Configuration

Each algorithm has its own config file:
- `configs/astar.yaml` - A* parameters (heuristic type, trajectory smoothing)
- `configs/dijkstra.yaml` - Dijkstra parameters
- `configs/rrt_star.yaml` - RRT* parameters (step_size, max_iterations, rewire_radius)
- `configs/apf.yaml` - APF parameters (zeta, eta, velocity limits)

## Output Files

Pre-generated demonstrations are available in `outputs/`:
- `outputs/astar/star_algorithm.gif` - A* with trajectory smoothing
- `outputs/dijkstra/dijkstra_algorithm.gif` - Dijkstra's algorithm
- `outputs/apf/apf_simulation.gif` - APF navigation
- `outputs/rrt_star/tree_expansion_with_rewire.gif` - RRT* tree growth
- `outputs/rrt_star/robot_simulation.gif` - Robot following RRT* path

## Algorithm Comparison

| Algorithm | Type | Optimal | Complexity | Best For |
|-----------|------|---------|------------|----------|
| A* | Graph | ✅ Yes | O(b^d) | Known environments, guaranteed optimality |
| Dijkstra | Graph | ✅ Yes | O(V²) | Shortest path without heuristic |
| RRT* | Sampling | ✅ Asymptotic | O(n log n) | High-dimensional spaces, complex constraints |
| APF | Reactive | ❌ No | O(1) per step | Real-time, dynamic environments |

## Adding New Algorithms

1. Create new file in `src/algorithms/`
2. Inherit from `PathPlanner` base class
3. Implement required methods: `plan()`, `get_metrics()`, `visualize()`
4. Add YAML config file in `configs/`
5. Register in `main.py`

Example:
```python
from src.core.path_planner import PathPlanner

class MyPlanner(PathPlanner):
    def _initialize_algorithm(self):
        # Setup algorithm-specific structures
        pass

    def plan(self, start, goal):
        # Implement planning logic
        pass

    def get_metrics(self):
        return {'algorithm': 'MyPlanner', ...}

    def visualize(self, ax, **kwargs):
        # Draw results
        pass
```

## Testing

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_astar.py
```

## Performance Benchmarks

Typical performance on 20x20 map with 3 obstacles:

| Algorithm | Path Length | Planning Time | Nodes Explored |
|-----------|-------------|---------------|----------------|
| A* | ~24.5 units | ~0.002s | ~15 nodes |
| Dijkstra | ~24.5 units | ~0.003s | ~25 nodes |
| RRT* | ~26.0 units | ~0.050s | 500 samples |
| APF | ~25.0 units | ~0.100s | ~800 steps |

## License

See root LICENSE file.

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- Docstrings for all public methods
- Unit tests for new features
- YAML configs for new parameters
