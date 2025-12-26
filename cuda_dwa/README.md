# CUDA-Accelerated Dynamic Window Approach (DWA)

GPU-parallelized implementation of the Dynamic Window Approach for real-time mobile robot navigation.

## Overview

This directory contains both GPU-accelerated (CUDA) and CPU baseline implementations of DWA, demonstrating 30-50x speedup through massive parallelization.

### What is DWA?

The **Dynamic Window Approach** is a velocity-based local path planning algorithm that:
1. Samples thousands of candidate (velocity, angular_velocity) pairs
2. Simulates each trajectory forward in time
3. Scores trajectories based on speed, heading, and obstacle avoidance
4. Selects the best collision-free trajectory

### Why CUDA?

- **Parallel Nature**: Each trajectory can be evaluated independently
- **GPU Advantage**: Evaluate 10,240 trajectories simultaneously instead of sequentially
- **Performance**: ~5-10ms on GPU vs ~300-500ms on CPU
- **Real-time Control**: Enables 100+ Hz planning for reactive navigation

## Directory Structure

```
cuda_dwa/
├── src/
│   ├── dwa_cuda.cu           # GPU implementation with CUDA kernels
│   ├── dwa_cpp.cpp            # CPU baseline for performance comparison
│   └── config_loader.cpp      # YAML configuration parser
├── include/
│   └── config_loader.h        # Configuration structures and headers
├── configs/                   # YAML configuration files
│   ├── robot.yaml             # Robot kinematic limits and initial pose
│   ├── environment.yaml       # Map parameters and goal position
│   ├── simulation.yaml        # Simulation timesteps and sampling
│   └── scoring.yaml           # Trajectory evaluation weights
├── outputs/
│   └── mobileapp.mp4          # Demo video
├── bin/                       # Compiled binaries (gitignored)
├── Makefile                   # Build system
└── README.md                  # This file
```

## Dependencies

### Required
- **CUDA Toolkit** (nvcc compiler)
  ```bash
  sudo apt-get install nvidia-cuda-toolkit
  ```

- **yaml-cpp library** (YAML parsing)
  ```bash
  sudo apt-get install libyaml-cpp-dev
  ```

- **g++ compiler** (C++ baseline)
  ```bash
  sudo apt-get install g++
  ```

### Check Dependencies
```bash
make check-deps
```

## Building

### Build Both Versions
```bash
make all
```

### Build Individual Versions
```bash
make cuda    # CUDA GPU version only
make cpp     # C++ baseline only
```

### Clean Build
```bash
make clean all
```

## Running

### Run CUDA Version
```bash
make run-cuda
```

### Run C++ Baseline
```bash
make run-cpp
```

### Performance Benchmark
```bash
make benchmark
```
This runs both versions sequentially and compares execution times.

## Configuration

All parameters are defined in YAML files (`configs/`). No recompilation needed for parameter changes!

### robot.yaml
```yaml
robot:
  max_linear_velocity: 2.0      # m/s
  max_angular_velocity: 3.14159 # rad/s (π)
  initial_pose:
    x: 10.0
    y: 10.0
    theta: 0.0
```

### environment.yaml
```yaml
environment:
  map:
    width: 1024           # Grid cells
    height: 1024
    resolution: 0.1       # Meters/cell
    obstacle_density: 0.05  # 5% obstacles
  goal:
    x: 90.0  # meters
    y: 90.0
```

### simulation.yaml
```yaml
simulation:
  dt: 0.1          # Timestep (seconds)
  sim_time: 3.0    # Simulation horizon
  num_samples: 10240  # Trajectories to evaluate
  cuda:
    block_size: 256  # Threads per block
```

### scoring.yaml
```yaml
scoring:
  speed_weight: 0.3     # Weight for forward velocity (30%)
  heading_weight: 0.7   # Weight for goal-directed motion (70%)
```

## Algorithm Details

### Kinematic Model (Differential Drive)

```
θ(t+dt) = θ(t) + ω·dt
x(t+dt) = x(t) + v·cos(θ)·dt
y(t+dt) = y(t) + v·sin(θ)·dt
```

Where:
- (x, y): Robot position
- θ: Heading angle
- v: Linear velocity
- ω: Angular velocity

### Scoring Function

```
score = (v / v_max) * 0.3 + (1 - dist_to_goal / 100) * 0.7
```

Components:
- **Speed**: Rewards faster motion (30% weight)
- **Heading**: Rewards motion toward goal (70% weight)
- **Collision**: Trajectories hitting obstacles get score = -∞

### CUDA Parallelization

- **Grid**: 40 blocks (10240 samples ÷ 256 threads/block)
- **Block**: 256 threads
- **Total threads**: 10,240 running simultaneously
- **Each thread**: Simulates one trajectory (~30 timesteps)

### Performance Characteristics

| Implementation | Time (ms) | Trajectories/sec | Speedup |
|----------------|-----------|------------------|---------|
| CPU (C++)      | 300-500   | ~25,000          | 1x      |
| GPU (CUDA)     | 5-10      | ~1,000,000       | 30-50x  |

## File Descriptions

### src/dwa_cuda.cu
- **evaluateTrajectories** kernel: Parallel trajectory simulation
- **reduceMax** kernel: Find best trajectory using tree reduction
- Comprehensive comments explaining CUDA optimization strategies

### src/dwa_cpp.cpp
- Sequential CPU baseline for comparison
- Identical algorithm to CUDA version
- Useful for debugging and verification

### src/config_loader.cpp
- Parses YAML configuration files
- Populates structs with robot/environment/simulation parameters
- Error handling for missing files or malformed YAML

## Usage Example

```bash
# 1. Adjust CUDA architecture in Makefile if needed
# (Current: sm_75 for RTX 2080)
vim Makefile  # Change CUDA_ARCH variable

# 2. Build both versions
make all

# 3. Run benchmark to compare CPU vs GPU
make benchmark

# 4. Modify parameters without rebuilding
vim configs/simulation.yaml  # Change num_samples to 20480

# 5. Re-run (instantly uses new config)
make run-cuda
```

## Extending the Code

### Add New Scoring Components

1. Edit `configs/scoring.yaml`:
   ```yaml
   scoring:
     speed_weight: 0.2
     heading_weight: 0.6
     clearance_weight: 0.2  # NEW
   ```

2. Update `include/config_loader.h`:
   ```cpp
   struct ScoringConfig {
       float speed_weight;
       float heading_weight;
       float clearance_weight;  // NEW
   };
   ```

3. Modify kernel in `src/dwa_cuda.cu`:
   ```cuda
   float clearance_score = min_obstacle_dist * clearance_weight;
   score += speed_score + heading_score + clearance_score;
   ```

### Tune for Your GPU

Adjust `CUDA_ARCH` in Makefile based on your GPU:
- RTX 2080/2070: 75
- RTX 3090/3080: 86
- RTX 4090: 89
- A100: 80

Find your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Common Issues

### Compilation Errors

**"nvcc: command not found"**
- Install CUDA Toolkit: `sudo apt-get install nvidia-cuda-toolkit`

**"yaml-cpp/yaml.h: No such file"**
- Install library: `sudo apt-get install libyaml-cpp-dev`

**"undefined reference to YAML::..."**
- Add `-lyaml-cpp` to linker flags (already in Makefile)

### Runtime Errors

**"CUDA error: invalid device function"**
- Wrong `CUDA_ARCH` in Makefile
- Set to your GPU's compute capability

**"no CUDA-capable device detected"**
- No NVIDIA GPU available
- Run C++ baseline instead: `make run-cpp`

**"Failed to load robot configuration"**
- YAML file missing or malformed
- Check `configs/` directory
- Validate YAML syntax

## Performance Tuning

### Increase Throughput
- Increase `num_samples` (e.g., 20480, 40960)
- GPU performance scales well to 100K+ samples

### Reduce Latency
- Decrease `num_samples` (e.g., 5120, 2560)
- Trade-off: Fewer samples may miss optimal trajectory

### Balance Quality vs Speed
```yaml
# High quality, slower (50ms)
num_samples: 40960
sim_time: 5.0
dt: 0.05

# Low quality, faster (2ms)
num_samples: 2560
sim_time: 2.0
dt: 0.2
```

## References

- Fox, D., Burgard, W., & Thrun, S. (1997). "The dynamic window approach to collision avoidance"
- NVIDIA CUDA Programming Guide: https://docs.nvidia.com/cuda/
- yaml-cpp Documentation: https://github.com/jbeder/yaml-cpp

## License

See root LICENSE file.
