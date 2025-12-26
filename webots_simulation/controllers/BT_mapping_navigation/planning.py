"""
A* Path Planning Behavior for Webots Robot.

This module implements a Behavior Tree node that computes an optimal collision-free
path from the robot's current position to a goal using the A* search algorithm.

Behavior Tree Characteristics:
- Status: Returns SUCCESS once path is computed
- Execution: Runs once after mapping completes
- Input: Reads occupancy grid map from blackboard (populated by mapping behavior)
- Output: Writes waypoint list to blackboard for navigation behavior

A* Algorithm Overview:
- Graph-based informed search algorithm
- Uses heuristic function (Euclidean distance) to guide search toward goal
- Guarantees optimal path if heuristic is admissible (never overestimates)
- Complexity: O(b^d) where b is branching factor, d is solution depth

Coordinate Systems:
- World Frame: Webots simulation coordinates (meters), origin at world center
- Map Frame: Discrete grid (pixels), 200x300 resolution
- Same transformation as mapping.py for consistency
"""

import py_trees
import numpy as np
from matplotlib import pyplot as plt

from heapq import heapify, heappush, heappop
from collections import defaultdict


def world2map(xw, yw):
    """
    Convert world coordinates to map grid coordinates.

    Coordinate System Details:
    - World Frame: Continuous space in meters
      - X-axis: Horizontal (left/right)
      - Y-axis: Vertical (forward/backward)
      - Origin: Center of simulation world

    - Map Frame: Discrete grid in pixels
      - Dimensions: 200 pixels (width) x 300 pixels (height)
      - Resolution: ~40 pixels/meter (X), ~50 pixels/meter (Y)

    Transformation Equations:
    - px = (xw + 2.25) * 40   → Shifts origin and scales to pixels
    - py = (yw - 2) * (-50)    → Shifts origin, scales, and inverts Y-axis

    Offset Explanation:
    - +2.25 in X: Shifts world origin to align with map grid
    - -2 in Y: Shifts world origin vertically
    - Negative scale in Y: Flips Y-axis (Webots uses +Y forward, map uses +Y down)

    Args:
        xw (float): X-coordinate in world frame (meters)
        yw (float): Y-coordinate in world frame (meters)

    Returns:
        list: [px, py] - Map coordinates in pixels, clamped to valid range

    Example:
        >>> world2map(0.0, 0.0)  # World center
        [90, 100]  # Approximate map center
    """
    # Apply offset and scaling
    px = int((xw + 2.25) * 40)
    py = int((yw - 2) * (-50))

    # Clamp to valid map bounds (0-199 for X, 0-299 for Y)
    px = min(px, 199)
    py = min(py, 299)
    px = max(px, 0)
    py = max(py, 0)

    return [px, py]


def map2world(px, py):
    """
    Convert map grid coordinates back to world coordinates.

    Inverse transformation of world2map().

    Args:
        px (int): X-coordinate in map frame (pixels)
        py (int): Y-coordinate in map frame (pixels)

    Returns:
        list: [xw, yw] - World coordinates in meters

    Example:
        >>> map2world(90, 100)
        [0.0, 0.0]  # Approximately world center
    """
    xw = px / 40 - 2.25
    yw = py / (-50) + 2
    return [xw, yw]


def getNeighbors(map, u):
    """
    Generate valid neighbors for A* search in 8-connected grid.

    Grid Connectivity:
    - 8-connected grid allows movement in 8 directions:
      - Cardinal: up, down, left, right (cost = 1.0)
      - Diagonal: up-left, up-right, down-left, down-right (cost = √2 ≈ 1.414)

    Movement Costs:
    - Horizontal/Vertical: distance = 1 pixel
    - Diagonal: distance = √(1² + 1²) = √2 pixels
    - This ensures accurate path length calculation

    Collision Checking:
    - Only returns cells with map value 0 (free space)
    - Cells with value 1 are obstacles (from C-space thresholding)

    Args:
        map (np.ndarray): Binary occupancy grid (200x300)
            - 0: Free space
            - 1: Obstacle
        u (tuple): Current cell (px, py)

    Returns:
        list: Valid neighbors as [(cost1, neighbor1), (cost2, neighbor2), ...]
            - cost: Movement cost (1.0 for cardinal, 1.414 for diagonal)
            - neighbor: Tuple (px, py) of neighboring cell

    Example:
        >>> getNeighbors(map, (50, 50))
        [(1.0, (50, 51)), (1.0, (50, 49)), ..., (1.414, (51, 51)), ...]
        # Returns up to 8 neighbors if all are free
    """
    neighbors = []

    # 8 possible movement directions:
    # (dx, dy): (right, up), (left, down), (horizontal), (vertical), (diagonals)
    for delta in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)):
        candidate = (u[0] + delta[0], u[1] + delta[1])

        # Check bounds and collision
        if (candidate[0] >= 0 and candidate[0] < len(map) and
            candidate[1] >= 0 and candidate[1] < len(map[0]) and
            map[candidate[0]][candidate[1]] == 0):

            # Calculate Euclidean distance as cost
            # Cardinal moves: sqrt(1^2 + 0^2) = 1.0
            # Diagonal moves: sqrt(1^2 + 1^2) = 1.414
            cost = np.sqrt(delta[0]**2 + delta[1]**2)
            neighbors.append((cost, candidate))

    return neighbors


class Planning(py_trees.behaviour.Behaviour):
    """
    A* path planning behavior.

    This behavior node computes an optimal collision-free path using the A* search
    algorithm on the occupancy grid map. The algorithm combines:
    1. Actual cost from start (g-score): Distance traveled so far
    2. Heuristic cost to goal (h-score): Estimated remaining distance
    3. Total cost (f-score): f = g + h

    A* Properties:
    - Optimal: Finds shortest path if heuristic is admissible
    - Complete: Always finds path if one exists
    - Efficient: Explores fewer nodes than Dijkstra by using heuristic

    Heuristic Function:
    - Euclidean distance: h(v) = √((goal_x - v_x)² + (goal_y - v_y)²)
    - Admissible: Never overestimates actual distance
    - Consistent: Satisfies triangle inequality

    Execution Flow:
    1. Read current robot position from GPS
    2. Convert start and goal to map coordinates
    3. Run A* search on occupancy grid
    4. Reconstruct path from parent pointers
    5. Convert path back to world coordinates
    6. Write waypoints to blackboard for navigation

    Attributes:
        robot: Webots Supervisor instance
        blackboard: Shared data storage
        world_goal (tuple): Goal position in world coordinates (meters)
        goal (tuple): Goal position in map coordinates (pixels)
        start (tuple): Start position in map coordinates (pixels)
        gps: GPS sensor device
        display: Robot display for visualization
    """

    def __init__(self, name, blackboard, dest):
        """
        Initialize the Planning behavior.

        Args:
            name (str): Behavior name for debugging
            blackboard: Shared blackboard for data exchange
            dest (tuple): Goal position in world coordinates (xw, yw)
        """
        super(Planning, self).__init__(name)
        self.robot = blackboard.read('robot')
        self.blackboard = blackboard

        # Store goal in both coordinate systems
        px, py = world2map(dest[0], dest[1])
        self.world_goal = dest  # World coordinates (meters)
        self.goal = (px, py)     # Map coordinates (pixels)
        self.start = None        # Will be set in initialise()

    def setup(self):
        """
        Initialize Webots devices.

        Called once during behavior tree setup. Acquires references to all
        required sensors and devices.
        """
        self.timestep = int(self.robot.getBasicTimeStep())

        # Display for path visualization
        self.display = self.robot.getDevice('display')

        # GPS for current position
        self.gps = self.robot.getDevice('gps')

        self.logger.debug("  %s [Planning::setup()]" % self.name)

    def initialise(self):
        """
        Get robot's current position as path starting point.

        Called once when the behavior first executes. Reads GPS to determine
        where the robot currently is, which becomes the A* search start node.
        """
        print("planning!")

        # Get current robot position from GPS
        xw = self.gps.getValues()[0]  # Robot X-position in world frame
        yw = self.gps.getValues()[1]  # Robot Y-position in world frame

        # Convert to map coordinates
        px, py = world2map(xw, yw)
        self.start = (px, py)

    def update(self):
        """
        Execute A* path planning algorithm.

        This method implements the complete A* search:

        Algorithm Pseudocode:
        1. Initialize open set (priority queue) with start node
        2. While open set not empty:
            a. Pop node u with lowest f-score
            b. If u is goal, reconstruct path and return
            c. For each neighbor v of u:
                - Calculate tentative g-score: g(u) + cost(u,v)
                - If better than previous g(v):
                    - Update g(v) and f(v) = g(v) + h(v)
                    - Add v to open set with priority f(v)
                    - Set parent(v) = u
        3. Reconstruct path by following parent pointers from goal to start

        Priority Queue:
        - Uses min-heap (heapq) with priority = f-score
        - f(v) = g(v) + h(v)
        - g(v): Actual cost from start to v
        - h(v): Heuristic (Euclidean distance from v to goal)

        Path Reconstruction:
        - Follow parent pointers backward from goal to start
        - Reverse to get start→goal path
        - Convert from map coordinates to world coordinates

        Returns:
            py_trees.common.Status.SUCCESS: Path computed and written to blackboard
        """
        # ============================================================
        # Step 1: Initialize A* Data Structures
        # ============================================================

        # Priority queue: stores (priority, node) tuples
        # Priority = f-score = g-score + heuristic
        queue = [(0, self.start)]
        heapify(queue)

        # Distance tracking: g-score for each node
        # g-score = actual cost from start to node
        distances = defaultdict(lambda: float('inf'))
        distances[self.start] = 0

        # Visited set: prevents re-exploring nodes
        visited = set()

        # Parent pointers: for path reconstruction
        # parent[v] = u means we reached v from u
        parent = {}

        # ============================================================
        # Step 2: A* Search Loop
        # ============================================================
        while queue:
            # Pop node with lowest f-score
            (priority, u) = heappop(queue)
            visited.add(u)

            # Check if goal reached
            if u == self.goal:
                break  # Path found, proceed to reconstruction

            # Explore neighbors
            for (costuv, v) in getNeighbors(self.blackboard.read('map'), u):
                if v not in visited:
                    # Calculate tentative g-score
                    # g(v) = g(u) + cost(u→v)
                    newcost = distances[u] + costuv

                    # Update if better path found
                    if newcost < distances[v]:
                        distances[v] = newcost

                        # Calculate f-score = g-score + heuristic
                        # Heuristic: Euclidean distance to goal
                        heuristic = np.sqrt((self.goal[0] - v[0])**2 + (self.goal[1] - v[1])**2)
                        f_score = newcost + heuristic

                        # Add to priority queue with f-score as priority
                        heappush(queue, (f_score, v))

                        # Record parent for path reconstruction
                        parent[v] = u

        # ============================================================
        # Step 3: Reconstruct Path
        # ============================================================
        path = []

        # Follow parent pointers backward from goal to start
        key = self.goal
        while key in parent.keys():
            key = parent[key]
            path.insert(0, key)  # Insert at front to reverse order

        # Add goal to complete the path
        path.append(self.goal)

        # ============================================================
        # Step 4: Visualize Path on Robot Display
        # ============================================================
        for (x, y) in path:
            self.display.setColor(0x00FFFF)  # Cyan color
            self.display.drawPixel(x, y)     # Draw path pixel

        # ============================================================
        # Step 5: Convert Path to World Coordinates
        # ============================================================
        # Navigation behavior needs waypoints in meters, not pixels
        converted_path = []
        for (px, py) in path:
            x, y = map2world(px, py)
            converted_path.append((x, y))
        path = converted_path

        # ============================================================
        # Step 6: Populate Blackboard with Waypoints
        # ============================================================
        # Navigation behavior will read 'waypoints' to follow the path
        self.blackboard.write('waypoints', path)

        return py_trees.common.Status.SUCCESS
