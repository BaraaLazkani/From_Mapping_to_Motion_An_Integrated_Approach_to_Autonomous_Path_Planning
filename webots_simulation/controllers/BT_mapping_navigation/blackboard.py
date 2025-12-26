"""
Blackboard Pattern for Behavior Tree Data Sharing.

This module implements a simple blackboard data structure for communication
between behavior tree nodes. The blackboard acts as shared memory, allowing
behaviors to exchange data without direct coupling.

Blackboard Pattern Overview
---------------------------

The blackboard pattern is a software design pattern where components communicate
through a shared data structure (the "blackboard") rather than direct method calls.

Key Characteristics:
1. **Decoupling**: Behaviors don't need references to each other
2. **Flexibility**: Easy to add new behaviors without modifying existing ones
3. **Simplicity**: Simple read/write interface
4. **Transparency**: All behaviors can see all data

Use Cases in This System:
- Robot reference: All behaviors need access to Webots robot object
- Map data: Mapping writes, Planning reads
- Waypoints: Planning writes, Navigation reads
- Exploration path: Main controller writes, Navigation reads

Alternative Approaches:
- Direct method calls: Tight coupling, hard to modify
- Message passing: More complex, harder to debug
- Global variables: Namespace pollution, hard to track

Why Blackboard for Behavior Trees?
- Natural fit: BT nodes are independent, blackboard provides data sharing
- py_trees library: Designed to work with blackboard pattern
- CSCA 5342 course: Standard pattern taught in robotics course

Example Data Flow:
1. Main controller writes 'robot' to blackboard
2. Mapping behavior reads 'robot', writes 'map' to blackboard
3. Planning behavior reads 'map', writes 'waypoints' to blackboard
4. Navigation behavior reads 'waypoints', follows path

Data Dictionary (Keys Used):
- 'robot': Webots Supervisor instance (read by all behaviors)
- 'waypoints': List of (x, y) tuples in world coordinates
  - Written by: Main controller (exploration), Planning (navigation)
  - Read by: Navigation
- 'map': 200x300 binary occupancy grid (0=free, 1=obstacle)
  - Written by: Mapping, DoesMapExist
  - Read by: Planning
"""


class Blackboard:
    """
    Simple dictionary-based blackboard for behavior tree data sharing.

    This class provides a minimal implementation of the blackboard pattern
    using a Python dictionary as the underlying storage. Behaviors can
    read and write data using string keys.

    Thread Safety:
    - NOT thread-safe (uses plain dict)
    - Sufficient for single-threaded Webots controller
    - For multi-threaded: use threading.Lock or py_trees.blackboard.Blackboard

    Data Lifetime:
    - Persists for entire simulation run
    - Cleared when simulation restarts
    - NOT saved to disk (unlike map in cspace.npy)

    Attributes:
        data (dict): Internal dictionary storing key-value pairs
    """

    def __init__(self):
        """
        Initialize empty blackboard.

        Creates an empty dictionary to store shared data. No predefined keys;
        behaviors create keys as needed via write() method.
        """
        self.data = {}

    def write(self, key, value):
        """
        Write data to blackboard.

        Stores or updates a key-value pair in the blackboard. If key already
        exists, its value is overwritten.

        Args:
            key (str): Identifier for the data (e.g., 'map', 'waypoints')
            value: Data to store (any Python object)

        Example:
            >>> blackboard.write('robot', robot_instance)
            >>> blackboard.write('map', numpy_array)
        """
        self.data[key] = value

    def read(self, key):
        """
        Read data from blackboard.

        Retrieves the value associated with the given key. Returns None if
        key doesn't exist (uses dict.get() with default=None).

        Args:
            key (str): Identifier for the data to retrieve

        Returns:
            Value associated with key, or None if key not found

        Example:
            >>> robot = blackboard.read('robot')
            >>> waypoints = blackboard.read('waypoints')
        """
        return self.data.get(key)
