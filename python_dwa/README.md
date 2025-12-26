# Python Dynamic Window Approach (DWA)

Real-time local path planning for differential drive robots using the Dynamic Window Approach algorithm with Pygame visualization.

## Overview

The Dynamic Window Approach (DWA) is a velocity-based local planning algorithm that evaluates multiple trajectory candidates in real-time to find optimal motion commands while avoiding obstacles. This implementation features:

- **Differential Drive Kinematics**: Complete model for two-wheeled robots
- **Dynamic Obstacle Avoidance**: Real-time response to moving obstacles
- **Trajectory Visualization**: Live display of evaluated paths
- **YAML Configuration**: All parameters externally configurable
- **GIF Recording**: Automatic generation of demonstration animations
- **Modular Architecture**: Clean separation of concerns for easy modification

## Algorithm Description

### Dynamic Window Approach

DWA operates in velocity space by:

1. **Dynamic Window Construction**: Compute reachable velocities given current velocities and acceleration limits
2. **Trajectory Sampling**: Generate trajectory predictions for sampled velocity pairs (vL, vR)
3. **Trajectory Evaluation**: Score each trajectory based on:
   - **Goal Approach**: Progress toward target position
   - **Obstacle Clearance**: Distance to nearest obstacle
4. **Velocity Selection**: Execute the velocity pair with highest score

### Differential Drive Kinematics

The robot uses a differential drive model with three motion types:

```
Straight Motion (vL = vR):
  x_new = x + v * Δt * cos(θ)
  y_new = y + v * Δt * sin(θ)
  θ_new = θ

Pure Rotation (vL = -vR):
  x_new = x
  y_new = y
  θ_new = θ + (vR - vL) * Δt / W

Arc Motion (general case):
  R = (W/2) * (vR + vL) / (vR - vL)
  x_new = x + R * (sin(Δθ + θ) - sin(θ))
  y_new = y - R * (cos(Δθ + θ) - cos(θ))
  θ_new = θ + Δθ
```

Where:
- `(x, y, θ)`: Robot pose (position and orientation)
- `vL, vR`: Left and right wheel velocities
- `W`: Wheel separation distance
- `R`: Turning radius
- `Δt`: Time step
- `Δθ = (vR - vL) * Δt / W`

### Scoring Function

Each trajectory is evaluated using:

```
benefit = forward_weight * progress - obstacle_weight * obstacle_penalty

progress = distance_previous - distance_new  (to target)
obstacle_penalty = max(0, safe_distance - closest_obstacle_distance)
```

## Directory Structure

```
python_dwa/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── robot.py              # Differential drive robot kinematics
│   ├── environment.py        # Obstacle management and collision detection
│   ├── dwa_planner.py        # Dynamic Window Approach algorithm
│   ├── visualizer.py         # Pygame visualization and GIF recording
│   ├── config_loader.py      # YAML configuration loading
│   ├── main.py               # Main entry point
│   └── DDWA.py               # Original implementation (preserved for reference)
├── configs/
│   └── dwa.yaml              # Configuration parameters
├── outputs/
│   └── simulation.gif        # Generated demonstration
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

```bash
# Python 3.7 or higher required
python3 --version
```

### Setup

```bash
# Navigate to python_dwa directory
cd python_dwa

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- `pygame>=2.0.0`: Real-time visualization
- `Pillow>=9.0.0`: GIF generation
- `numpy>=1.21.0`: Numerical computations
- `PyYAML>=6.0`: Configuration file loading

## Usage

### Basic Execution

```bash
# Run with default configuration
cd src
python main.py

# Run with custom configuration
python main.py --config path/to/custom_config.yaml
```

### Configuration

All parameters are defined in `configs/dwa.yaml`:

```yaml
robot:
  radius: 0.10                # Robot radius (meters)
  wheel_width: 0.20           # Wheel separation (meters)
  max_velocity: 0.3           # Maximum wheel velocity (m/s)
  max_acceleration: 0.5       # Maximum acceleration (m/s²)
  safe_distance: 0.10         # Minimum safe distance from obstacles (m)

planning:
  timestep: 0.1               # Simulation timestep (seconds)
  steps_ahead: 10             # Planning horizon (steps)
  forward_weight: 12          # Weight for goal approach
  obstacle_weight: 6666       # Penalty for obstacle proximity

environment:
  playfield:
    x_min: -4.0
    x_max: 4.0
    y_min: -3.0
    y_max: 3.0
  obstacles:
    barrier_radius: 0.1       # Dynamic obstacle radius (m)
    num_dynamic: 35           # Number of moving obstacles
    velocity_range: 0.1       # Max obstacle velocity (m/s)
    static:                   # Static large obstacles
      - {x: -2.0, y: 1.0}
      - {x: 2.0, y: 1.0}
      - {x: 0.0, y: 2.0}

visualization:
  width: 1500                 # Window width (pixels)
  height: 1000                # Window height (pixels)
  scale: 160                  # Pixels per meter
  record_gif: true            # Enable GIF recording
  gif_duration: 15            # Recording duration (seconds)
  frame_skip: 5               # Capture every Nth frame
```

## Code Architecture

### Module Descriptions

#### `robot.py` - Robot Kinematics
- **Class**: `DifferentialDriveRobot`
- **Purpose**: Implements differential drive kinematic model
- **Key Methods**:
  - `predict_position()`: Forward kinematics simulation
  - `update()`: Update robot state
  - `get_pose()`: Query current position and orientation

#### `environment.py` - Environment Management
- **Class**: `Environment`
- **Purpose**: Manages obstacles and collision detection
- **Key Methods**:
  - `generate_dynamic_obstacles()`: Create moving obstacles
  - `update_obstacles()`: Update obstacle positions
  - `calculate_closest_obstacle_distance()`: Collision checking
  - `check_target_reached()`: Goal detection

#### `dwa_planner.py` - DWA Algorithm
- **Class**: `DWAPlanner`
- **Purpose**: Implements Dynamic Window Approach
- **Key Methods**:
  - `plan()`: Execute one planning cycle
  - `_evaluate_trajectory()`: Score trajectory candidates

#### `visualizer.py` - Visualization
- **Class**: `Visualizer`
- **Purpose**: Pygame rendering and GIF recording
- **Key Methods**:
  - `render()`: Draw complete scene
  - `_draw_robot()`: Render robot with differential drive wheels
  - `_draw_trajectories()`: Visualize evaluated paths

#### `config_loader.py` - Configuration
- **Functions**:
  - `load_config()`: Load YAML configuration
  - `get_robot_params()`, `get_planning_params()`, etc.: Extract parameter groups

#### `main.py` - Entry Point
- **Function**: `main()`
- **Purpose**: Initialize and run simulation loop

## Visualization

The simulation window displays:

- **Robot**: White circle with two blue wheels showing differential drive
- **Target Obstacle**: Light blue circle (robot's goal)
- **Dynamic Obstacles**: Dark red circles (moving, avoiding static obstacles)
- **Static Obstacles**: Large gray circles (stationary hazards)
- **Robot Trail**: Faint white dots showing path history
- **Planned Trajectories**: Green lines/arcs showing DWA-evaluated paths

## Performance

Typical performance on modern hardware:
- **Planning Frequency**: ~100 Hz (10ms planning cycles)
- **Trajectory Evaluations**: 9 velocity pairs per cycle
- **Obstacle Handling**: 35+ dynamic obstacles in real-time
- **Visualization**: 20 FPS (adjustable via `time.sleep()`)

## GIF Recording

When `record_gif: true` in configuration:
1. Simulation runs for `gif_duration` seconds
2. Every `frame_skip`-th frame is captured
3. Frames are scaled to 50% resolution
4. GIF saved to `outputs/simulation.gif`
5. Recording stops automatically

Adjust `frame_skip` to balance file size vs. smoothness:
- `frame_skip: 1` → Smooth but large file
- `frame_skip: 5` → Moderate size, slight jerkiness
- `frame_skip: 10` → Small file, noticeable skipping

## Comparison with CUDA Implementation

| Feature | Python DWA | CUDA DWA |
|---------|-----------|----------|
| **Implementation** | Sequential CPU | Parallel GPU |
| **Planning Time** | ~10ms (9 trajectories) | ~5ms (10,240 trajectories) |
| **Trajectory Sampling** | 3×3 grid | Dense sampling |
| **Use Case** | Development, debugging, visualization | Real-time high-performance control |
| **Ease of Modification** | Very easy | Moderate (CUDA knowledge required) |
| **Dependencies** | Pygame, NumPy | CUDA toolkit, yaml-cpp |

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'yaml'"
**Solution**: Install dependencies: `pip install -r requirements.txt`

### Issue: Simulation runs too fast
**Solution**: Increase `time.sleep()` value in `main.py:134`

### Issue: GIF file too large
**Solution**: Increase `frame_skip` in `configs/dwa.yaml` (e.g., from 5 to 10)

### Issue: Robot gets stuck near obstacles
**Solution**: Adjust weights in config:
- Increase `obstacle_weight` for more cautious behavior
- Increase `safe_distance` for wider safety margins

### Issue: Robot doesn't reach target
**Solution**:
- Increase `forward_weight` to prioritize goal approach
- Reduce obstacle density (`num_dynamic`)

## Extensions and Modifications

### Add New Obstacle Types
1. Modify `environment.py:Environment` class
2. Add new obstacle generation in `generate_dynamic_obstacles()`
3. Update collision detection in `calculate_closest_obstacle_distance()`

### Change Scoring Function
1. Edit `dwa_planner.py:DWAPlanner._evaluate_trajectory()`
2. Add new terms (e.g., energy efficiency, smoothness)
3. Update weights in `configs/dwa.yaml`

### Modify Visualization
1. Edit `visualizer.py:Visualizer` class
2. Add new rendering in `render()` or create new `_draw_*()` methods
3. Adjust colors in `__init__()`

## References

- **Original DWA Paper**: Fox, D., Burgard, W., & Thrun, S. (1997). "The dynamic window approach to collision avoidance"
- **Differential Drive Kinematics**: Siegwart, R., & Nourbakhsh, I. R. (2004). "Introduction to Autonomous Mobile Robots"

## License

See repository root [LICENSE](../LICENSE) file.

## Authors

- **Baraa Lazkani** - [BaraaLazkani](https://github.com/BaraaLazkani)
- **Modar Ibrahim**
- **Laith Alsheikh**

All authors contributed equally to this work.

## Citation

If you use this code, please cite:

```bibtex
@misc{lazkani2024autonomous,
  title={From Mapping to Motion: An Integrated Approach to Autonomous Path Planning},
  author={Lazkani, Baraa and Ibrahim, Modar and Alsheikh, Laith},
  year={2024},
  publisher={GitHub},
  url={https://github.com/BaraaLazkani/From_Mapping_to_Motion_An_Integrated_Approach_to_Autonomous_Path_Planning}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## See Also

- **Python Algorithms** (`../python_algorithms/`): Global path planning (A*, Dijkstra, RRT*, APF)
- **CUDA DWA** (`../cuda_dwa/`): GPU-accelerated version with 30-50x speedup
- **Webots Simulation** (`../webots_simulation/`): Complete SLAM + navigation pipeline
