/**
 * @file config_loader.cpp
 * @brief Implementation of YAML configuration loader for CUDA DWA.
 *
 * This file implements functions to parse YAML configuration files using
 * the yaml-cpp library. Each function reads a specific configuration file
 * and populates the corresponding struct.
 *
 * Error Handling:
 * - Missing files: Exits with error message
 * - Missing keys: Uses default values (yaml-cpp returns 0/false for missing keys)
 * - Invalid types: yaml-cpp throws exception, caught and reported
 *
 * YAML-CPP Library Usage:
 * - YAML::LoadFile(): Reads YAML file into YAML::Node object
 * - node["key"]: Accesses nested keys
 * - node.as<T>(): Converts YAML value to C++ type T
 */

#include "config_loader.h"
#include <yaml-cpp/yaml.h>
#include <iostream>
#include <stdexcept>


RobotConfig loadRobotConfig(const std::string& filepath) {
    try {
        YAML::Node config = YAML::LoadFile(filepath);
        RobotConfig robot;

        // Parse kinematic constraints
        robot.max_linear_velocity = config["robot"]["max_linear_velocity"].as<float>();
        robot.max_angular_velocity = config["robot"]["max_angular_velocity"].as<float>();

        // Parse initial pose
        robot.initial_pose.x = config["robot"]["initial_pose"]["x"].as<float>();
        robot.initial_pose.y = config["robot"]["initial_pose"]["y"].as<float>();
        robot.initial_pose.theta = config["robot"]["initial_pose"]["theta"].as<float>();

        return robot;
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading robot config: " << e.what() << std::endl;
        throw std::runtime_error("Failed to load robot configuration");
    }
}


EnvironmentConfig loadEnvironmentConfig(const std::string& filepath) {
    try {
        YAML::Node config = YAML::LoadFile(filepath);
        EnvironmentConfig env;

        // Parse map parameters
        env.map.width = config["environment"]["map"]["width"].as<int>();
        env.map.height = config["environment"]["map"]["height"].as<int>();
        env.map.resolution = config["environment"]["map"]["resolution"].as<float>();
        env.map.obstacle_density = config["environment"]["map"]["obstacle_density"].as<float>();

        // Parse goal position
        env.goal.x = config["environment"]["goal"]["x"].as<float>();
        env.goal.y = config["environment"]["goal"]["y"].as<float>();

        return env;
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading environment config: " << e.what() << std::endl;
        throw std::runtime_error("Failed to load environment configuration");
    }
}


SimulationConfig loadSimulationConfig(const std::string& filepath) {
    try {
        YAML::Node config = YAML::LoadFile(filepath);
        SimulationConfig sim;

        // Parse simulation parameters
        sim.dt = config["simulation"]["dt"].as<float>();
        sim.sim_time = config["simulation"]["sim_time"].as<float>();
        sim.num_samples = config["simulation"]["num_samples"].as<int>();

        // Parse CUDA execution parameters
        sim.cuda.block_size = config["simulation"]["cuda"]["block_size"].as<int>();

        return sim;
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading simulation config: " << e.what() << std::endl;
        throw std::runtime_error("Failed to load simulation configuration");
    }
}


ScoringConfig loadScoringConfig(const std::string& filepath) {
    try {
        YAML::Node config = YAML::LoadFile(filepath);
        ScoringConfig scoring;

        // Parse scoring weights
        scoring.speed_weight = config["scoring"]["speed_weight"].as<float>();
        scoring.heading_weight = config["scoring"]["heading_weight"].as<float>();

        return scoring;
    } catch (const YAML::Exception& e) {
        std::cerr << "Error loading scoring config: " << e.what() << std::endl;
        throw std::runtime_error("Failed to load scoring configuration");
    }
}


DWAConfig loadDWAConfig(const std::string& config_dir) {
    /**
     * Load all configuration files from specified directory.
     *
     * Expected directory structure:
     * config_dir/
     * ├── robot.yaml
     * ├── environment.yaml
     * ├── simulation.yaml
     * └── scoring.yaml
     */

    DWAConfig config;

    // Construct file paths
    std::string robot_path = config_dir + "/robot.yaml";
    std::string environment_path = config_dir + "/environment.yaml";
    std::string simulation_path = config_dir + "/simulation.yaml";
    std::string scoring_path = config_dir + "/scoring.yaml";

    // Load each configuration file
    std::cout << "Loading DWA configuration from: " << config_dir << std::endl;

    config.robot = loadRobotConfig(robot_path);
    std::cout << "  ✓ Loaded robot configuration" << std::endl;

    config.environment = loadEnvironmentConfig(environment_path);
    std::cout << "  ✓ Loaded environment configuration" << std::endl;

    config.simulation = loadSimulationConfig(simulation_path);
    std::cout << "  ✓ Loaded simulation configuration" << std::endl;

    config.scoring = loadScoringConfig(scoring_path);
    std::cout << "  ✓ Loaded scoring configuration" << std::endl;

    // Validate configuration
    if (config.scoring.speed_weight + config.scoring.heading_weight > 1.01f ||
        config.scoring.speed_weight + config.scoring.heading_weight < 0.99f) {
        std::cerr << "Warning: Scoring weights do not sum to 1.0" << std::endl;
    }

    std::cout << "Configuration loaded successfully!" << std::endl;
    return config;
}
