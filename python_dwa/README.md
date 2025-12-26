# Python Dynamic Window Approach (DWA)

Real-time local path planning for differential drive robots with Pygame visualization.

## Overview

The Dynamic Window Approach (DWA) is a velocity-based local planning algorithm that evaluates multiple trajectory candidates to find the optimal motion command while avoiding obstacles.

## Features

- Real-time DWA planning with differential drive kinematics
- Dynamic obstacle avoidance
- Pygame visualization with trajectory display
- GIF recording for demonstration
- YAML configuration support

## Directory Structure

```
python_dwa/
├── src/
│   └── DDWA.py            # Original DWA implementation (to be refactored)
├── configs/
│   └── dwa.yaml           # Configuration parameters
├── outputs/
│   └── simulation.gif     # Demo animation
├── requirements.txt
└── README.md
```

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run original implementation
cd "Final Codes/Python Codes/DWA code"
python DDWA.py
```

## Configuration

Edit `configs/dwa.yaml` to adjust robot parameters, planning weights, and visualization settings.

## Algorithm Details

### Kinematic Model

Differential drive robot:
- Straight motion: vL = vR
- Pure rotation: vL = -vR
- Arc motion: General case with radius R

### DWA Planning

1. Sample velocity pairs (vL, vR) within dynamic window
2. Simulate each trajectory forward
3. Score based on:
   - Forward progress toward goal
   - Distance to nearest obstacle
4. Select best trajectory and execute

## Output

- `simulation.gif`: Recorded animation of robot navigation
- Trajectory visualization showing evaluated paths

## See Also

- CUDA DWA (`cuda_dwa/`): GPU-accelerated version
- Python algorithms (`python_algorithms/`): Other planning algorithms
