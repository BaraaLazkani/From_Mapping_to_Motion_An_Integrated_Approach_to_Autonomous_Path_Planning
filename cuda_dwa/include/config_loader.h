/**
 * @file config_loader.h
 * @brief YAML Configuration Loader for CUDA DWA
 *
 * This header defines configuration structures and loading functions for the
 * CUDA-accelerated Dynamic Window Approach (DWA) planner.
 *
 * Configuration System:
 * - Replaces hardcoded constants with YAML files
 * - Enables parameter tuning without recompilation
 * - Supports multiple configuration profiles (testing, deployment)
 *
 * Dependencies:
 * - yaml-cpp library: sudo apt-get install libyaml-cpp-dev
 * - Link with: -lyaml-cpp
 */

#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <string>


/**
 * @struct RobotConfig
 * @brief Robot kinematic constraints and initial pose.
 *
 * Defines the robot's physical limitations and starting configuration.
 */
struct RobotConfig {
    float max_linear_velocity;   ///< Maximum forward velocity (m/s)
    float max_angular_velocity;  ///< Maximum rotation rate (rad/s)

    struct {
        float x;      ///< Initial X position (meters)
        float y;      ///< Initial Y position (meters)
        float theta;  ///< Initial heading angle (radians)
    } initial_pose;
};


/**
 * @struct EnvironmentConfig
 * @brief Occupancy grid map parameters and goal position.
 *
 * Defines the environment representation for collision checking and navigation.
 */
struct EnvironmentConfig {
    struct {
        int width;              ///< Map width in cells
        int height;             ///< Map height in cells
        float resolution;       ///< Meters per cell
        float obstacle_density; ///< Fraction of cells with obstacles (for testing)
    } map;

    struct {
        float x;  ///< Goal X position (meters)
        float y;  ///< Goal Y position (meters)
    } goal;
};


/**
 * @struct SimulationConfig
 * @brief Trajectory simulation and CUDA execution parameters.
 *
 * Controls how trajectories are simulated and how CUDA kernels are launched.
 */
struct SimulationConfig {
    float dt;              ///< Time step for forward simulation (seconds)
    float sim_time;        ///< Total simulation horizon (seconds)
    int num_samples;       ///< Number of (velocity, omega) pairs to evaluate

    struct {
        int block_size;    ///< CUDA threads per block (typically 256)
    } cuda;
};


/**
 * @struct ScoringConfig
 * @brief Trajectory evaluation weights.
 *
 * Defines how trajectories are scored for selection. Weights should sum to 1.0.
 */
struct ScoringConfig {
    float speed_weight;    ///< Weight for forward velocity component (0-1)
    float heading_weight;  ///< Weight for goal-directed motion (0-1)
};


/**
 * @struct DWAConfig
 * @brief Complete DWA configuration aggregating all sub-configs.
 *
 * Combines robot, environment, simulation, and scoring configurations into
 * a single structure for easy parameter passing.
 */
struct DWAConfig {
    RobotConfig robot;
    EnvironmentConfig environment;
    SimulationConfig simulation;
    ScoringConfig scoring;
};


/**
 * @brief Load all DWA configuration from YAML files.
 *
 * Reads four YAML configuration files and constructs a complete DWAConfig.
 * If any file is missing or has parsing errors, the program exits with an
 * error message.
 *
 * @param config_dir Path to directory containing YAML files (default: "configs")
 * @return DWAConfig Complete configuration struct
 *
 * @throws std::runtime_error if YAML files are missing or malformed
 *
 * Example Usage:
 * @code
 *   DWAConfig config = loadDWAConfig("cuda_dwa/configs");
 *   std::cout << "Max speed: " << config.robot.max_linear_velocity << std::endl;
 * @endcode
 */
DWAConfig loadDWAConfig(const std::string& config_dir = "configs");


/**
 * @brief Load robot configuration from YAML file.
 *
 * @param filepath Path to robot.yaml
 * @return RobotConfig Parsed robot configuration
 */
RobotConfig loadRobotConfig(const std::string& filepath);


/**
 * @brief Load environment configuration from YAML file.
 *
 * @param filepath Path to environment.yaml
 * @return EnvironmentConfig Parsed environment configuration
 */
EnvironmentConfig loadEnvironmentConfig(const std::string& filepath);


/**
 * @brief Load simulation configuration from YAML file.
 *
 * @param filepath Path to simulation.yaml
 * @return SimulationConfig Parsed simulation configuration
 */
SimulationConfig loadSimulationConfig(const std::string& filepath);


/**
 * @brief Load scoring configuration from YAML file.
 *
 * @param filepath Path to scoring.yaml
 * @return ScoringConfig Parsed scoring configuration
 */
ScoringConfig loadScoringConfig(const std::string& filepath);

#endif // CONFIG_LOADER_H
