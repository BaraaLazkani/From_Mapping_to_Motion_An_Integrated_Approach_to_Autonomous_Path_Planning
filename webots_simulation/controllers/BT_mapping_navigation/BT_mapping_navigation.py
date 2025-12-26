"""
Behavior Tree Controller for Autonomous Robot SLAM and Navigation.

This is the main Webots robot controller that orchestrates the complete autonomous
navigation pipeline using a Behavior Tree architecture. The system performs:

1. **SLAM (Simultaneous Localization and Mapping)**:
   - LiDAR-based occupancy grid mapping (while exploring)
   - Map persistence (save/load from disk)

2. **Path Planning**:
   - A* algorithm on occupancy grid
   - Optimal collision-free paths

3. **Navigation**:
   - Differential drive PD control
   - Waypoint following

Behavior Tree Architecture
--------------------------

Behavior Trees provide a modular, hierarchical control structure with clear
execution semantics. Key advantages:
- **Modularity**: Each behavior is independent
- **Readability**: Tree structure matches task decomposition
- **Reactivity**: Can interrupt and switch behaviors
- **Reusability**: Behaviors can be composed in different trees

Node Types Used:
1. **Sequence**: Execute children in order, fail if any child fails
   - Returns RUNNING while executing children
   - Returns FAILURE if any child fails
   - Returns SUCCESS when all children succeed

2. **Selector**: Try children in order until one succeeds
   - Returns RUNNING while executing current child
   - Returns SUCCESS if any child succeeds
   - Returns FAILURE if all children fail

3. **Parallel**: Execute multiple children concurrently
   - Policy: SuccessOnOne → succeed when first child succeeds
   - Used for simultaneous mapping and exploration

Memory Management:
- memory=True: Resume from last running child on next tick
- Without memory: Restart from first child every tick
- Critical for long-running behaviors (mapping, navigation)

Execution Flow
--------------

Main Sequence (executes sequentially):
├─ Selector: "Does map exist?"
│  ├─ DoesMapExist: Check if cspace.npy exists on disk
│  │                 → SUCCESS if exists (skip mapping)
│  │                 → FAILURE if not exists (proceed to mapping)
│  │
│  └─ Parallel: "Mapping" (if map doesn't exist)
│     ├─ Mapping: Build occupancy grid from LiDAR
│     │           Status: RUNNING (continuous mapping)
│     └─ Navigation: Explore environment with predefined waypoints
│                    Status: RUNNING → SUCCESS when done
│
│  Policy: SuccessOnOne → parallel stops when navigation completes
│  Result: Map is built and saved during exploration
│
├─ Planning: Compute path to lower left corner (-1.46, -3.12)
│            Status: SUCCESS when path computed
│
├─ Navigation: Follow path to lower left corner
│              Status: RUNNING → SUCCESS when arrived
│
├─ Planning: Compute path to sink (0.88, 0.09)
│            Status: SUCCESS when path computed
│
└─ Navigation: Follow path to sink
               Status: RUNNING → SUCCESS when arrived

Complete Task:
1. Check if map exists
2. If not: Explore environment while building map
3. Plan path to corner
4. Navigate to corner
5. Plan path to sink
6. Navigate to sink

Robot Configuration
-------------------

Platform: TurtleBot3 Burger (Differential Drive)

Sensors:
- GPS: Position feedback (xw, yw, zw)
- Compass: Orientation feedback (heading angle)
- LiDAR: Hokuyo URG-04LX-UG01
  - Range: ~5.6 meters
  - FOV: 240 degrees (4.19 radians)
  - Rays: 667 total (507 used after filtering)

Actuators:
- Left wheel motor: wheel_left_joint
- Right wheel motor: wheel_right_joint
- Max velocity: 6.28 rad/s (2π rad/s)

Coordinate System:
- World frame: Webots simulation coordinates (meters)
- Map frame: 200x300 occupancy grid (pixels)
- Origin offset: See mapping.py and planning.py for transformations
"""

# Core scientific computing
import numpy as np
from matplotlib import pyplot as plt

# Behavior Tree library
import py_trees
from py_trees.composites import Sequence, Parallel, Selector

# Custom behavior implementations
from navigation import Navigation
from database import DoesMapExist
from mapping import Mapping
from planning import Planning
from blackboard import Blackboard

# Webots simulator API
from controller import Supervisor


# ============================================================
# Initialize Robot
# ============================================================
# Supervisor allows access to all robot devices and simulation state
robot = Supervisor()

# Get simulation timestep (typically 32ms)
timestep = int(robot.getBasicTimeStep())


# ============================================================
# Define Exploration Waypoints
# ============================================================
# Predefined waypoints for initial mapping exploration
# These guide the robot around the environment to build a complete map
# Waypoints are in world coordinates (xw, yw) in meters
WP = [(0.614, -0.19),   # Start near table
      (0.77, -0.94),    # Move along table
      (0.37, -3.04),    # Explore lower area
      (-1.41, -3.39),   # Lower left region
      (-1.53, -3.39),   # Continue lower left
      (-1.8, -1.46),    # Move up left side
      (-1.44, 0.38),    # Upper left region
      (0.0, 0.0)]       # Return to origin


# ============================================================
# Initialize Sensors (Enable Before BT Execution)
# ============================================================
# CRITICAL: All sensors must be enabled before entering behavior tree
# Otherwise: First sensor readings will be invalid/zero

# GPS sensor: Provides position feedback
gps = robot.getDevice('gps')
gps.enable(timestep)

# Compass sensor: Provides orientation feedback
compass = robot.getDevice('compass')
compass.enable(timestep)

# LiDAR sensor: Hokuyo URG-04LX-UG01
lidar = robot.getDevice('Hokuyo URG-04LX-UG01')
lidar.enable(timestep)
lidar.enablePointCloud()  # Enable 3D point cloud (we'll use 2D slice)


# ============================================================
# Initialize Motors (Velocity Control Mode)
# ============================================================
# CRITICAL: Set position to infinity to switch from position to velocity control
# Without this: Motors default to position control mode

leftMotor = robot.getDevice('wheel_left_joint')
rightMotor = robot.getDevice('wheel_right_joint')
leftMotor.setPosition(float('inf'))   # Enable velocity control mode
rightMotor.setPosition(float('inf'))  # Enable velocity control mode


# ============================================================
# Initialize Blackboard (Shared Data Storage)
# ============================================================
# Blackboard provides shared memory for behavior communication
# Behaviors read/write data without direct coupling

blackboard = Blackboard()

# Store robot reference for all behaviors
blackboard.write('robot', robot)

# Create exploration path: original waypoints + reverse (round trip)
# This ensures thorough environment coverage during mapping
# np.flip(WP, 0) reverses waypoint order → robot returns to start
exploration_path = np.concatenate((WP, np.flip(WP, 0)), axis=0)
blackboard.write('waypoints', exploration_path)


# ============================================================
# Construct Behavior Tree
# ============================================================
# Tree structure implements the complete mission logic
# memory=True ensures behaviors resume from where they left off

tree = Sequence("Main", children=[
    # --------------------------------------------------------
    # Stage 1: Map Acquisition (Load or Build)
    # --------------------------------------------------------
    # Selector tries children until one succeeds
    # If map exists → DoesMapExist succeeds, skip mapping
    # If map missing → DoesMapExist fails, proceed to Parallel mapping
    Selector("Does map exist?", children=[
        # Check if map file exists on disk
        DoesMapExist("Test for map", blackboard),

        # If map doesn't exist: Build it via parallel mapping & exploration
        Parallel("Mapping", policy=py_trees.common.ParallelPolicy.SuccessOnOne(), children=[
            # Continuously build occupancy grid from LiDAR
            # Status: RUNNING (never completes on its own)
            Mapping("map the environment", blackboard),

            # Follow exploration waypoints to cover environment
            # Status: RUNNING → SUCCESS when all waypoints reached
            Navigation("move around the table", blackboard)
        ])
        # Parallel policy: SuccessOnOne
        # - When Navigation completes → Parallel succeeds
        # - Mapping terminate() is called → saves map to disk
    ], memory=True),

    # --------------------------------------------------------
    # Stage 2: Navigate to Lower Left Corner
    # --------------------------------------------------------
    # Goal: (-1.46, -3.12) in world coordinates
    # Planning reads map from blackboard, computes A* path
    Planning("compute path to lower left corner", blackboard, (-1.46, -3.12)),

    # Navigation reads waypoints from blackboard, follows path
    Navigation("move to lower left corner", blackboard),

    # --------------------------------------------------------
    # Stage 3: Navigate to Sink
    # --------------------------------------------------------
    # Goal: (0.88, 0.09) in world coordinates
    Planning("compute path to sink", blackboard, (0.88, 0.09)),

    # Navigate to final goal
    Navigation("move to sink", blackboard)
], memory=True)


# ============================================================
# Setup Behavior Tree
# ============================================================
# Invoke setup() method on all nodes in the tree
# This initializes device references and prepares behaviors for execution
tree.setup_with_descendants()


# ============================================================
# Main Simulation Loop
# ============================================================
# Step through Webots simulation and tick behavior tree

while robot.step(timestep) != -1:
    # Advance simulation by one timestep (32ms)
    # Returns -1 when simulation ends or is stopped

    # Tick behavior tree once per timestep
    # - Resumes from last running node (memory=True)
    # - Propagates status up the tree
    # - Triggers initialise() on newly started behaviors
    # - Triggers terminate() on completed/interrupted behaviors
    tree.tick_once()
