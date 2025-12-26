/**
 * @file dwa_cpp.cpp
 * @brief CPU Baseline Implementation of Dynamic Window Approach (DWA)
 *
 * This file provides a sequential C++ implementation of the Dynamic Window
 * Approach for mobile robot navigation. It serves as a performance baseline
 * for comparison with the GPU-accelerated CUDA version (dwa_cuda.cu).
 *
 * Purpose of This Baseline
 * ------------------------
 *
 * 1. **Performance Comparison**: Measure CPU vs GPU speedup
 *    - Typical speedup: 30-50x for CUDA implementation
 *    - Helps justify GPU acceleration effort
 *
 * 2. **Algorithm Verification**: Ensure correctness
 *    - Both versions should select same trajectory (given same inputs)
 *    - Easier to debug sequential code than parallel CUDA
 *
 * 3. **Fallback Implementation**: GPU not always available
 *    - Some systems lack CUDA-capable GPU
 *    - Provides functional alternative
 *
 * Sequential vs Parallel Execution
 * --------------------------------
 *
 * CPU (this file):
 * - Evaluates trajectories sequentially (one at a time)
 * - Single-threaded: for loop iterating over all samples
 * - Execution time: O(num_samples × num_simulation_steps)
 * - Typical time: 300-500ms for 10,240 trajectories
 *
 * GPU (dwa_cuda.cu):
 * - Evaluates all trajectories in parallel
 * - Multi-threaded: 10,240 CUDA threads simultaneously
 * - Execution time: ~constant (limited by simulation depth)
 * - Typical time: 5-10ms for 10,240 trajectories
 *
 * Algorithm Equivalence
 * ---------------------
 *
 * Both implementations use identical:
 * - Kinematic model (differential drive)
 * - Collision detection (occupancy grid lookup)
 * - Scoring function (speed + heading weights)
 * - Simulation parameters (dt, sim_time, num_samples)
 *
 * Only difference is execution model (sequential vs parallel).
 *
 * Code Organization
 * -----------------
 *
 * - ObstacleMap struct: Encapsulates occupancy grid and collision checking
 * - Trajectory struct: Stores evaluation results (velocity, score, collision)
 * - evaluateTrajectory(): Simulates one trajectory (called in loop)
 * - main(): Orchestrates full DWA pipeline
 *
 * Compilation
 * -----------
 *
 * Requires:
 * - C++ compiler (g++, clang++)
 * - yaml-cpp library (libyaml-cpp-dev)
 *
 * Compile:
 *   g++ -o dwa_cpp src/dwa_cpp.cpp src/config_loader.cpp \
 *       -I include -lyaml-cpp -std=c++14
 *
 * Run:
 *   ./dwa_cpp
 *
 * @author DWA C++ Baseline
 * @date 2024
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <ctime>

#include "config_loader.h"


/**
 * @struct ObstacleMap
 * @brief Occupancy grid representation for collision detection.
 *
 * Encapsulates a 2D occupancy grid and provides methods for collision
 * checking. The grid is stored as a flattened 1D array (row-major order).
 *
 * Grid Representation:
 * - 0.0: Free space (robot can traverse)
 * - 1.0: Obstacle (robot cannot traverse)
 *
 * Coordinate Systems:
 * - World coordinates: (x, y) in meters
 * - Grid coordinates: (grid_x, grid_y) in cells
 * - Conversion: grid_x = floor(x / resolution)
 */
struct ObstacleMap {
    int width;         ///< Map width in cells
    int height;        ///< Map height in cells
    float resolution;  ///< Meters per grid cell
    std::vector<float> data;  ///< Flattened occupancy grid (row-major)

    /**
     * @brief Constructor initializes map and randomly places obstacles.
     *
     * Generates a random obstacle map for testing purposes. In a real
     * application, this would load a map from a file or SLAM system.
     *
     * @param w Map width in cells
     * @param h Map height in cells
     * @param res Map resolution (meters/cell)
     * @param obstacle_density Fraction of cells to mark as obstacles
     */
    ObstacleMap(int w, int h, float res, float obstacle_density)
        : width(w), height(h), resolution(res), data(w * h, 0.0f)
    {
        // Randomly place obstacles
        int num_obstacles = obstacle_density * width * height;
        for (int i = 0; i < num_obstacles; ++i) {
            int x = std::rand() % width;
            int y = std::rand() % height;
            data[y * width + x] = 1.0f;
        }
    }

    /**
     * @brief Check if a world-coordinate position collides with an obstacle.
     *
     * Converts world coordinates to grid coordinates and checks the
     * occupancy grid. Returns true if:
     * - Position is outside map bounds, OR
     * - Grid cell contains obstacle (value ≥ 0.5)
     *
     * @param x World X coordinate (meters)
     * @param y World Y coordinate (meters)
     * @return true if collision, false if free space
     */
    bool isObstacle(float x, float y) const {
        // Convert world coordinates to grid coordinates
        int grid_x = static_cast<int>(x / resolution);
        int grid_y = static_cast<int>(y / resolution);

        // Check bounds
        if (grid_x < 0 || grid_x >= width ||
            grid_y < 0 || grid_y >= height) {
            return true;  // Out of bounds = collision
        }

        // Check occupancy grid (row-major indexing)
        return data[grid_y * width + grid_x] >= 0.5f;
    }
};


/**
 * @struct Trajectory
 * @brief Stores the result of evaluating a single DWA trajectory.
 *
 * Contains the velocity command, final score, and collision status
 * for one evaluated trajectory.
 */
struct Trajectory {
    float velocity;   ///< Linear velocity (m/s)
    float omega;      ///< Angular velocity (rad/s)
    float score;      ///< Trajectory score (higher is better, -∞ if collision)
    bool collision;   ///< True if trajectory collided with obstacle
};


/**
 * @brief Evaluate a single DWA trajectory.
 *
 * Simulates the robot forward in time using the given velocity command,
 * checking for collisions and computing a trajectory score.
 *
 * Simulation Loop:
 * 1. Update robot pose using kinematic model
 * 2. Check for collision with obstacles
 * 3. Compute incremental score
 * 4. Repeat for simulation horizon
 *
 * Kinematic Model (Differential Drive):
 * - θ(t+dt) = θ(t) + ω·dt
 * - x(t+dt) = x(t) + v·cos(θ)·dt
 * - y(t+dt) = y(t) + v·sin(θ)·dt
 *
 * Scoring:
 * - Speed: (v / v_max) * speed_weight
 * - Heading: (1 - dist_to_goal / 100) * heading_weight
 * - Collision: score = -∞
 *
 * @param velocity Linear velocity command (m/s)
 * @param omega Angular velocity command (rad/s)
 * @param map Obstacle map for collision detection
 * @param curr_x Current robot X position (meters)
 * @param curr_y Current robot Y position (meters)
 * @param curr_theta Current robot heading (radians)
 * @param goal_x Goal X position (meters)
 * @param goal_y Goal Y position (meters)
 * @param max_speed Maximum linear velocity (for normalization)
 * @param dt Simulation timestep (seconds)
 * @param sim_time Total simulation horizon (seconds)
 * @param speed_weight Weight for speed component
 * @param heading_weight Weight for heading component
 * @return Trajectory Result struct with velocity, score, collision status
 */
Trajectory evaluateTrajectory(
    float velocity, float omega, const ObstacleMap& map,
    float curr_x, float curr_y, float curr_theta,
    float goal_x, float goal_y,
    float max_speed, float dt, float sim_time,
    float speed_weight, float heading_weight)
{
    // Initialize simulation state from current robot pose
    float x = curr_x;
    float y = curr_y;
    float theta = curr_theta;

    float score = 0.0f;      // Accumulated score
    bool collision = false;  // Collision flag

    // ============================================================
    // Forward Simulate Trajectory
    // ============================================================
    // Integrate from t=0 to t=sim_time with timestep dt
    for (float t = 0; t < sim_time; t += dt) {
        // Update heading
        theta += omega * dt;

        // Update position
        x += velocity * cosf(theta) * dt;
        y += velocity * sinf(theta) * dt;

        // Check for collision
        if (map.isObstacle(x, y)) {
            collision = true;
            break;  // Stop simulating this trajectory
        }

        // Compute distance to goal
        float dist_to_goal = hypotf(goal_x - x, goal_y - y);

        // Compute trajectory score components
        float speed_score = (velocity / max_speed) * speed_weight;
        float heading_score = (1.0f - (dist_to_goal / 100.0f)) * heading_weight;

        // Accumulate score
        score += speed_score + heading_score;
    }

    // Return trajectory evaluation result
    return {velocity, omega, collision ? -INFINITY : score, collision};
}


/**
 * @brief Main entry point for C++ baseline DWA.
 *
 * Workflow:
 * 1. Load configuration from YAML files
 * 2. Initialize random velocity samples
 * 3. Generate random obstacle map
 * 4. Sequentially evaluate all trajectories
 * 5. Find best trajectory
 * 6. Print results and performance metrics
 */
int main() {
    // ============================================================
    // Step 1: Load Configuration from YAML Files
    // ============================================================
    DWAConfig config = loadDWAConfig("configs");

    // Extract configuration values for readability
    const int map_width = config.environment.map.width;
    const int map_height = config.environment.map.height;
    const float map_resolution = config.environment.map.resolution;
    const float obstacle_density = config.environment.map.obstacle_density;

    const float curr_x = config.robot.initial_pose.x;
    const float curr_y = config.robot.initial_pose.y;
    const float curr_theta = config.robot.initial_pose.theta;

    const float goal_x = config.environment.goal.x;
    const float goal_y = config.environment.goal.y;

    const int num_samples = config.simulation.num_samples;
    const float max_speed = config.robot.max_linear_velocity;
    const float max_omega = config.robot.max_angular_velocity;
    const float dt = config.simulation.dt;
    const float sim_time = config.simulation.sim_time;

    const float speed_weight = config.scoring.speed_weight;
    const float heading_weight = config.scoring.heading_weight;

    // ============================================================
    // Step 2: Generate Random Velocity Samples
    // ============================================================
    std::srand(std::time(0));  // Seed random number generator

    std::vector<float> velocities(num_samples);
    std::vector<float> omegas(num_samples);

    // Sample uniformly from velocity space
    for (int i = 0; i < num_samples; ++i) {
        // Linear velocity: v ∈ [0, max_speed]
        velocities[i] = static_cast<float>(rand()) / RAND_MAX * max_speed;

        // Angular velocity: ω ∈ [-max_omega, max_omega]
        omegas[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * max_omega;
    }

    // ============================================================
    // Step 3: Generate Random Obstacle Map
    // ============================================================
    ObstacleMap map(map_width, map_height, map_resolution, obstacle_density);

    // ============================================================
    // Step 4: Start CPU Timing
    // ============================================================
    auto start = std::chrono::high_resolution_clock::now();

    // ============================================================
    // Step 5: Sequentially Evaluate All Trajectories
    // ============================================================
    // Initialize best trajectory with invalid score
    Trajectory best_trajectory{0, 0, -INFINITY, false};

    // Sequential loop: evaluate one trajectory at a time
    // This is the key difference from CUDA (which evaluates all in parallel)
    for (int i = 0; i < num_samples; ++i) {
        // Evaluate this (velocity, omega) trajectory
        Trajectory traj = evaluateTrajectory(
            velocities[i], omegas[i], map,
            curr_x, curr_y, curr_theta,
            goal_x, goal_y,
            max_speed, dt, sim_time,
            speed_weight, heading_weight
        );

        // Update best trajectory if this one is better
        if (traj.score > best_trajectory.score) {
            best_trajectory = traj;
        }
    }

    // ============================================================
    // Step 6: Stop CPU Timing
    // ============================================================
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = stop - start;

    // ============================================================
    // Step 7: Print Results
    // ============================================================
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "C++ Baseline DWA Results" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << "Best trajectory:" << std::endl;
    std::cout << "  Linear velocity:  " << best_trajectory.velocity << " m/s" << std::endl;
    std::cout << "  Angular velocity: " << best_trajectory.omega << " rad/s" << std::endl;
    std::cout << "  Score: " << best_trajectory.score << std::endl;

    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Trajectories evaluated: " << num_samples << std::endl;
    std::cout << "  CPU execution time: " << duration.count() << " ms" << std::endl;
    std::cout << "  Throughput: " << (num_samples / duration.count() * 1000.0) << " trajectories/sec" << std::endl;

    std::cout << std::string(60, '=') << std::endl;

    return 0;
}
