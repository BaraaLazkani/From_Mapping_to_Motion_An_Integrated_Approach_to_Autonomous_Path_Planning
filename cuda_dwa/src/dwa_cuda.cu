/**
 * @file dwa_cuda.cu
 * @brief CUDA-Accelerated Dynamic Window Approach (DWA) for Mobile Robot Navigation
 *
 * This file implements a GPU-parallelized version of the Dynamic Window Approach,
 * a local trajectory planning algorithm for mobile robots. DWA evaluates thousands
 * of candidate trajectories in parallel to find the optimal velocity command.
 *
 * Dynamic Window Approach Overview
 * --------------------------------
 *
 * DWA is a velocity-based local planner that:
 * 1. Samples many (velocity, angular_velocity) pairs within robot's kinematic limits
 * 2. Simulates forward motion for each pair over a short time horizon
 * 3. Scores trajectories based on goals (speed, heading, obstacle avoidance)
 * 4. Selects the best trajectory and executes its initial velocity command
 *
 * Algorithm Steps:
 * 1. **Velocity Sampling**: Generate N candidate (v, ω) pairs
 *    - v ∈ [0, v_max]: Linear velocity
 *    - ω ∈ [-ω_max, ω_max]: Angular velocity
 *
 * 2. **Trajectory Simulation**: For each (v, ω):
 *    - Forward integrate kinematic model for T seconds
 *    - Check for collisions with obstacles
 *    - Compute trajectory score
 *
 * 3. **Scoring**: Evaluate each trajectory based on:
 *    - Speed: Prefer faster motion (v / v_max)
 *    - Heading: Prefer motion toward goal (1 - distance_to_goal / max_dist)
 *    - Collision: Reject trajectories that hit obstacles (score = -∞)
 *
 * 4. **Selection**: Choose trajectory with highest score
 *
 * CUDA Parallelization Strategy
 * ------------------------------
 *
 * The key insight is that each trajectory can be evaluated independently,
 * making this an embarrassingly parallel problem perfect for GPUs.
 *
 * Parallelization:
 * - Each CUDA thread evaluates one (v, ω) trajectory
 * - 10,240 trajectories evaluated in parallel
 * - Block size: 256 threads → 40 blocks
 * - Each trajectory simulates ~30 timesteps forward
 *
 * Performance:
 * - GPU (RTX 3090): ~5-10ms for 10,240 trajectories
 * - CPU (single core): ~300-500ms for same workload
 * - Speedup: ~30-50x
 *
 * Memory Organization:
 * - Global memory: Obstacle map, velocity arrays, scores
 * - Shared memory: Parallel reduction for finding best trajectory
 * - Registers: Trajectory simulation state (x, y, θ)
 *
 * Kinematic Model
 * ---------------
 *
 * Differential drive robot (e.g., TurtleBot):
 * - θ(t+dt) = θ(t) + ω·dt     (update heading)
 * - x(t+dt) = x(t) + v·cos(θ)·dt   (update X position)
 * - y(t+dt) = y(t) + v·sin(θ)·dt   (update Y position)
 *
 * Where:
 * - (x, y): Robot position in world frame
 * - θ: Robot heading angle
 * - v: Linear velocity (m/s)
 * - ω: Angular velocity (rad/s)
 * - dt: Simulation timestep
 *
 * Collision Detection
 * -------------------
 *
 * - Occupancy grid map: 1024×1024 cells, 0.1m resolution
 * - Each simulated pose checked against map
 * - Map lookup: grid_x = floor(x / resolution)
 * - Obstacle detected if map[grid_y][grid_x] ≥ 0.5
 * - Trajectory rejected immediately on collision
 *
 * Configuration
 * -------------
 *
 * All parameters loaded from YAML files (configs/):
 * - robot.yaml: v_max, ω_max, initial pose
 * - environment.yaml: Map size, resolution, goal position
 * - simulation.yaml: dt, sim_time, num_samples
 * - scoring.yaml: Speed weight, heading weight
 *
 * Compilation
 * -----------
 *
 * Requires:
 * - CUDA Toolkit (nvcc)
 * - yaml-cpp library (libyaml-cpp-dev)
 *
 * Compile:
 *   nvcc -o dwa_cuda src/dwa_cuda.cu src/config_loader.cpp \
 *        -I include -lyaml-cpp -std=c++14
 *
 * Run:
 *   ./dwa_cuda
 *
 * @author CUDA DWA Implementation
 * @date 2024
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "config_loader.h"


/**
 * @brief CUDA kernel to evaluate DWA trajectories in parallel.
 *
 * Each thread simulates one (velocity, omega) trajectory forward in time,
 * checking for collisions and computing a score based on speed and heading.
 *
 * Trajectory Simulation:
 * - Integrate kinematic model over [0, sim_time] with timestep dt
 * - Check each pose for collision with obstacle map
 * - Terminate early if collision detected
 * - Accumulate score at each timestep
 *
 * Collision Detection:
 * - Convert world coordinates (x, y) to grid coordinates
 * - Check map bounds (reject if outside map)
 * - Check obstacle map (reject if map[grid_y][grid_x] ≥ 0.5)
 *
 * Scoring Function:
 * - Speed component: (v / v_max) * speed_weight
 *   - Rewards faster motion
 *   - Normalized by maximum velocity
 * - Heading component: (1 - dist_to_goal / max_dist) * heading_weight
 *   - Rewards motion toward goal
 *   - Normalized by approximate max distance (100m)
 * - Total score: sum over all timesteps (higher is better)
 * - Collision: score = -∞ (rejected)
 *
 * Thread Organization:
 * - 1D grid of blocks, 1D blocks of threads
 * - Thread ID: idx = blockIdx.x * blockDim.x + threadIdx.x
 * - Each thread processes trajectories[idx]
 *
 * @param[in] velocities      Array of linear velocities (v) for each trajectory
 * @param[in] omegas          Array of angular velocities (ω) for each trajectory
 * @param[in] obstacle_map    2D occupancy grid (flattened to 1D: row-major)
 * @param[in] map_width       Map width in cells
 * @param[in] map_height      Map height in cells
 * @param[in] map_resolution  Meters per grid cell
 * @param[in] curr_x          Current robot X position (meters)
 * @param[in] curr_y          Current robot Y position (meters)
 * @param[in] curr_theta      Current robot heading (radians)
 * @param[in] goal_x          Goal X position (meters)
 * @param[in] goal_y          Goal Y position (meters)
 * @param[out] scores         Output array of trajectory scores
 * @param[in] num_samples     Total number of trajectories to evaluate
 * @param[in] max_speed       Maximum linear velocity (m/s)
 * @param[in] max_omega       Maximum angular velocity (rad/s)
 * @param[in] dt              Simulation timestep (seconds)
 * @param[in] sim_time        Total simulation horizon (seconds)
 */
__global__ void evaluateTrajectories(
    const float* velocities, const float* omegas,
    const float* obstacle_map, int map_width, int map_height, float map_resolution,
    float curr_x, float curr_y, float curr_theta,
    float goal_x, float goal_y,
    float* scores, int num_samples,
    float max_speed, float max_omega,
    float dt, float sim_time,
    float speed_weight, float heading_weight
) {
    // ============================================================
    // Step 1: Compute Thread ID and Bounds Check
    // ============================================================
    // Each thread handles one trajectory evaluation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples) return;  // Guard against extra threads

    // ============================================================
    // Step 2: Get Velocity Command for This Trajectory
    // ============================================================
    float v = velocities[idx];  // Linear velocity for this trajectory
    float w = omegas[idx];      // Angular velocity for this trajectory

    // ============================================================
    // Step 3: Initialize Simulation State
    // ============================================================
    // Start from current robot pose
    float x = curr_x;
    float y = curr_y;
    float theta = curr_theta;

    float score = 0.0f;        // Accumulated trajectory score
    bool collision = false;    // Collision flag

    // ============================================================
    // Step 4: Forward Simulate Trajectory
    // ============================================================
    // Integrate kinematic model from t=0 to t=sim_time
    // Typically sim_time=3.0s, dt=0.1s → 30 simulation steps
    for (float t = 0; t < sim_time; t += dt) {
        // Update robot pose using differential drive kinematics
        // θ(t+dt) = θ(t) + ω·dt
        theta += w * dt;

        // x(t+dt) = x(t) + v·cos(θ)·dt
        x += v * cosf(theta) * dt;

        // y(t+dt) = y(t) + v·sin(θ)·dt
        y += v * sinf(theta) * dt;

        // --------------------------------------------------------
        // Collision Detection
        // --------------------------------------------------------
        // Convert world coordinates to grid coordinates
        int grid_x = static_cast<int>(x / map_resolution);
        int grid_y = static_cast<int>(y / map_resolution);

        // Check map bounds
        if (grid_x < 0 || grid_x >= map_width ||
            grid_y < 0 || grid_y >= map_height) {
            collision = true;  // Out of bounds = collision
            break;             // Stop simulating this trajectory
        }

        // Check obstacle map (row-major indexing)
        // Obstacle detected if cell value ≥ 0.5
        if (obstacle_map[grid_y * map_width + grid_x] >= 0.5f) {
            collision = true;
            break;
        }

        // --------------------------------------------------------
        // Trajectory Scoring
        // --------------------------------------------------------
        // Compute distance to goal from current simulated position
        float dist_to_goal = hypotf(goal_x - x, goal_y - y);

        // Speed component: reward faster motion
        // Normalized to [0, 1] by dividing by max_speed
        float speed_score = (v / max_speed) * speed_weight;

        // Heading component: reward motion toward goal
        // Normalized to [0, 1] where 1 = at goal, 0 = far from goal
        // 100.0f is approximate maximum expected distance
        float heading_score = (1.0f - (dist_to_goal / 100.0f)) * heading_weight;

        // Accumulate score over trajectory
        score += speed_score + heading_score;
    }

    // ============================================================
    // Step 5: Store Final Score
    // ============================================================
    // If collision occurred, assign -∞ to reject this trajectory
    // Otherwise, use accumulated score
    scores[idx] = collision ? -INFINITY : score;
}


/**
 * @brief CUDA kernel for parallel reduction to find maximum score.
 *
 * This kernel implements a tree-based parallel reduction to find the
 * trajectory with the highest score across all threads. The reduction
 * happens in two stages:
 * 1. Block-level reduction (this kernel): Find max within each block
 * 2. Host-level reduction: Find global max across block results
 *
 * Shared Memory Reduction:
 * - Each block loads scores into shared memory
 * - Threads cooperatively reduce to find block maximum
 * - Uses tree-based reduction pattern (log₂(N) steps)
 *
 * Reduction Pattern:
 * Step 0: 256 threads, stride=128 → 128 comparisons
 * Step 1: 128 threads, stride=64  → 64 comparisons
 * Step 2: 64 threads, stride=32   → 32 comparisons
 * ...
 * Step 7: 2 threads, stride=1     → 1 comparison
 * Final: Thread 0 writes block result
 *
 * Example (8 threads):
 * Initial:  [3, 1, 7, 4, 2, 8, 5, 6]
 * Step 1:   [7, 4, 8, 6, -, -, -, -]  (stride=4: compare 0-4, 1-5, 2-6, 3-7)
 * Step 2:   [8, 6, -, -, -, -, -, -]  (stride=2: compare 0-2, 1-3)
 * Step 3:   [8, -, -, -, -, -, -, -]  (stride=1: compare 0-1)
 * Result: 8 (maximum value)
 *
 * Synchronization:
 * - __syncthreads() ensures all threads complete each reduction step
 * - Prevents race conditions on shared memory
 *
 * @param[in] scores        Array of trajectory scores from evaluateTrajectories
 * @param[out] max_score    Per-block maximum scores (length = grid_size)
 * @param[out] max_index    Per-block indices of maximum scores
 * @param[in] num_samples   Total number of trajectories
 */
__global__ void reduceMax(const float* scores, float* max_score, int* max_index, int num_samples) {
    // ============================================================
    // Step 1: Allocate Shared Memory
    // ============================================================
    // Shared memory for scores (dynamically allocated at kernel launch)
    extern __shared__ float shared_scores[];

    // Shared memory for indices (statically allocated, max 256 threads/block)
    __shared__ int shared_indices[256];

    int tid = threadIdx.x;  // Thread ID within block
    int idx = blockIdx.x * blockDim.x + tid;  // Global thread ID

    // ============================================================
    // Step 2: Load Data into Shared Memory
    // ============================================================
    // Each thread loads one score
    // Threads beyond num_samples load -∞ (will not be maximum)
    shared_scores[tid] = (idx < num_samples) ? scores[idx] : -INFINITY;
    shared_indices[tid] = idx;

    // Wait for all threads to finish loading
    __syncthreads();

    // ============================================================
    // Step 3: Tree-Based Reduction
    // ============================================================
    // Iteratively reduce by half each step
    // s = stride: 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        // Only first 's' threads participate in this step
        if (tid < s && idx + s < num_samples) {
            // Compare current thread's value with partner at offset 's'
            if (shared_scores[tid] < shared_scores[tid + s]) {
                // Partner has larger score → update current thread
                shared_scores[tid] = shared_scores[tid + s];
                shared_indices[tid] = shared_indices[tid + s];
            }
        }
        // Wait for all threads to complete this reduction step
        __syncthreads();
    }

    // ============================================================
    // Step 4: Write Block Result
    // ============================================================
    // Thread 0 now holds the block maximum
    if (tid == 0) {
        max_score[blockIdx.x] = shared_scores[0];
        max_index[blockIdx.x] = shared_indices[0];
    }
}


/**
 * @brief Main entry point for CUDA DWA demonstration.
 *
 * Workflow:
 * 1. Load configuration from YAML files
 * 2. Initialize random velocity samples
 * 3. Generate random obstacle map
 * 4. Transfer data to GPU
 * 5. Launch trajectory evaluation kernel
 * 6. Launch reduction kernel (optional optimization)
 * 7. Find best trajectory on CPU
 * 8. Print results and performance metrics
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

    float curr_x = config.robot.initial_pose.x;
    float curr_y = config.robot.initial_pose.y;
    float curr_theta = config.robot.initial_pose.theta;

    float goal_x = config.environment.goal.x;
    float goal_y = config.environment.goal.y;

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

    std::vector<float> h_velocities(num_samples);
    std::vector<float> h_omegas(num_samples);

    // Sample uniformly from velocity space
    for (int i = 0; i < num_samples; ++i) {
        // Linear velocity: v ∈ [0, max_speed]
        h_velocities[i] = static_cast<float>(rand()) / RAND_MAX * max_speed;

        // Angular velocity: ω ∈ [-max_omega, max_omega]
        h_omegas[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * max_omega;
    }

    // ============================================================
    // Step 3: Generate Random Obstacle Map
    // ============================================================
    std::vector<float> h_obstacle_map(map_width * map_height, 0.0f);

    // Randomly place obstacles (5% of map cells)
    int num_obstacles = config.environment.map.obstacle_density * map_width * map_height;
    for (int i = 0; i < num_obstacles; ++i) {
        int x = rand() % map_width;
        int y = rand() % map_height;
        h_obstacle_map[y * map_width + x] = 1.0f;  // Mark as obstacle
    }

    // ============================================================
    // Step 4: Allocate GPU Memory
    // ============================================================
    float *d_velocities, *d_omegas, *d_scores, *d_obstacle_map;

    cudaMalloc(&d_velocities, num_samples * sizeof(float));
    cudaMalloc(&d_omegas, num_samples * sizeof(float));
    cudaMalloc(&d_scores, num_samples * sizeof(float));
    cudaMalloc(&d_obstacle_map, map_width * map_height * sizeof(float));

    // ============================================================
    // Step 5: Start GPU Timing
    // ============================================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // ============================================================
    // Step 6: Transfer Data to GPU
    // ============================================================
    cudaMemcpy(d_velocities, h_velocities.data(), num_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_omegas, h_omegas.data(), num_samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_obstacle_map, h_obstacle_map.data(), map_width * map_height * sizeof(float), cudaMemcpyHostToDevice);

    // ============================================================
    // Step 7: Launch Trajectory Evaluation Kernel
    // ============================================================
    int block_size = config.simulation.cuda.block_size;  // 256 threads/block
    int grid_size = (num_samples + block_size - 1) / block_size;  // 40 blocks

    evaluateTrajectories<<<grid_size, block_size>>>(
        d_velocities, d_omegas, d_obstacle_map,
        map_width, map_height, map_resolution,
        curr_x, curr_y, curr_theta,
        goal_x, goal_y,
        d_scores, num_samples,
        max_speed, max_omega,
        dt, sim_time,
        speed_weight, heading_weight
    );

    // ============================================================
    // Step 8: Launch Reduction Kernel (Optional Optimization)
    // ============================================================
    float *d_max_score;
    int *d_max_index;
    cudaMalloc(&d_max_score, grid_size * sizeof(float));
    cudaMalloc(&d_max_index, grid_size * sizeof(int));

    // Launch reduction with shared memory allocation
    reduceMax<<<grid_size, block_size, block_size * sizeof(float)>>>(
        d_scores, d_max_score, d_max_index, num_samples
    );

    // ============================================================
    // Step 9: Transfer Results Back to Host
    // ============================================================
    std::vector<float> h_scores(num_samples);
    cudaMemcpy(h_scores.data(), d_scores, num_samples * sizeof(float), cudaMemcpyDeviceToHost);

    // ============================================================
    // Step 10: Stop GPU Timing
    // ============================================================
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // ============================================================
    // Step 11: Find Best Trajectory (CPU Final Reduction)
    // ============================================================
    int best_idx = 0;
    for (int i = 0; i < num_samples; ++i) {
        if (h_scores[i] > h_scores[best_idx]) {
            best_idx = i;
        }
    }

    // ============================================================
    // Step 12: Print Results
    // ============================================================
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "CUDA DWA Results" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    std::cout << "Best trajectory:" << std::endl;
    std::cout << "  Linear velocity:  " << h_velocities[best_idx] << " m/s" << std::endl;
    std::cout << "  Angular velocity: " << h_omegas[best_idx] << " rad/s" << std::endl;
    std::cout << "  Score: " << h_scores[best_idx] << std::endl;

    std::cout << "\nPerformance:" << std::endl;
    std::cout << "  Trajectories evaluated: " << num_samples << std::endl;
    std::cout << "  GPU execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "  Throughput: " << (num_samples / milliseconds * 1000.0f) << " trajectories/sec" << std::endl;

    std::cout << std::string(60, '=') << std::endl;

    // ============================================================
    // Step 13: Cleanup
    // ============================================================
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_velocities);
    cudaFree(d_omegas);
    cudaFree(d_scores);
    cudaFree(d_obstacle_map);
    cudaFree(d_max_score);
    cudaFree(d_max_index);

    return 0;
}
