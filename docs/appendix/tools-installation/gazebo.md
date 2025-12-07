# Installation Guide: Gazebo (Garden)

## Overview

This guide provides detailed instructions for installing Gazebo Garden, the latest version of the Gazebo simulation environment. Gazebo is essential for the digital twin and simulation modules in this book.

## System Requirements

### Minimum Requirements
- **Operating System**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Architecture**: 64-bit (amd64)
- **RAM**: 8 GB (16 GB recommended for complex simulations)
- **GPU**: Dedicated GPU with OpenGL 3.3+ support (NVIDIA/AMD recommended)
- **Disk Space**: 5 GB free space
- **Processor**: Multi-core processor (Intel Core i5 or equivalent AMD)

### Recommended Requirements
- **Operating System**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Architecture**: 64-bit (amd64)
- **RAM**: 32 GB or more for complex multi-robot simulations
- **GPU**: NVIDIA RTX series or AMD Radeon Pro with 8GB+ VRAM
- **Disk Space**: 20 GB free space for simulation assets
- **Processor**: Multi-core processor (Intel Core i7 or equivalent AMD)

## Pre-Installation Steps

### 1. System Update
Update your system packages to ensure compatibility:

```bash
sudo apt update
sudo apt upgrade
```

### 2. Install Prerequisites
Install essential system packages:

```bash
sudo apt install wget lsb-release gnupg
```

### 3. Verify GPU Support
Check that your system has proper GPU support:

```bash
lspci | grep -E "VGA|3D"
nvidia-smi  # If using NVIDIA GPU
```

## Installation Process

### 1. Add Gazebo Repository
Add the Gazebo package repository to your system:

```bash
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/gazebo-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gazebo-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
```

### 2. Update Package Lists
```bash
sudo apt update
```

### 3. Install Gazebo Garden
Install the complete Gazebo Garden suite:

```bash
sudo apt install gz-garden
```

### 4. Install Additional Gazebo Tools
Install additional tools and libraries:

```bash
sudo apt install libgz-sim8-dev libgz-common5-dev libgz-fuel-tools9-dev
sudo apt install libgz-physics6-dev libgz-math8-dev libgz-tools2-dev
```

### 5. Install ROS 2 Gazebo Integration
Install packages for ROS 2 integration:

```bash
sudo apt install ros-humble-gazebo-ros ros-humble-gazebo-ros-pkgs
sudo apt install ros-humble-gazebo-dev
```

## Environment Setup

### 1. Source Gazebo Environment
Gazebo requires environment variables to function properly:

```bash
source /usr/share/gz/setup.sh
```

### 2. Permanent Environment Setup
Add Gazebo setup to your bashrc:

```bash
echo "source /usr/share/gz/setup.sh" >> ~/.bashrc
```

### 3. Verify Installation
Test the installation:

```bash
gz --versions
```

You should see the installed Gazebo Garden version.

## Testing Gazebo

### 1. Launch Basic Simulation
```bash
gz sim
```

This should open the Gazebo GUI with a default empty world.

### 2. Load Sample World
```bash
gz sim -r -v 1 shapes.sdf
```

### 3. Test Command Line Interface
```bash
gz topic -l  # List available topics
```

## Development Environment Setup

### 1. Create Gazebo Workspace
Create a workspace for custom models and worlds:

```bash
mkdir -p ~/gazebo_ws/models
mkdir -p ~/gazebo_ws/worlds
mkdir -p ~/gazebo_ws/plugins
```

### 2. Set Environment Variables
Add to your bashrc for custom model paths:

```bash
echo 'export GZ_SIM_RESOURCE_PATH="~/gazebo_ws/models:$GZ_SIM_RESOURCE_PATH"' >> ~/.bashrc
echo 'export GZ_SIM_WORLD_PATH="~/gazebo_ws/worlds:$GZ_SIM_WORLD_PATH"' >> ~/.bashrc
```

### 3. Install Model Libraries
Install additional model libraries:

```bash
sudo apt install gz-tools
```

## Plugin Development Setup

### 1. Install Development Libraries
```bash
sudo apt install build-essential cmake pkg-config
sudo apt install libgz-sim8-dev libsdformat14-dev
```

### 2. Install Physics Libraries
```bash
sudo apt install libgz-physics6-dev libgz-common5-dev
```

## Troubleshooting Common Issues

### 1. GPU/OpenGL Issues
If you encounter rendering issues:
```bash
# Check OpenGL support
glxinfo | grep "OpenGL version"
# Ensure proper drivers are installed
sudo apt install mesa-utils
```

### 2. Performance Issues
For better performance:
- Ensure dedicated GPU is being used
- Close unnecessary applications
- Increase system RAM if possible

### 3. Missing Models
If models don't appear:
```bash
# Check model path
echo $GZ_SIM_RESOURCE_PATH
# Download models from Fuel (Gazebo's online model repository)
```

### 4. Permission Issues
If you have permission issues:
```bash
sudo chown -R $USER:$USER ~/gazebo_ws
```

## Advanced Configuration

### 1. Custom Configuration
Create custom configuration file:
```bash
mkdir -p ~/.gz/sim
# Add custom settings to ~/.gz/sim/gz_sim_gui.conf
```

### 2. Physics Engine Configuration
Gazebo supports multiple physics engines. Configure in world files:
- DART (default)
- Bullet
- ODE
- Simbody

### 3. Rendering Engine
Gazebo supports multiple rendering engines:
- OpenGL (default)
- Vulkan (if available)

## Integration with ROS 2

### 1. Install ROS 2 Bridge
```bash
sudo apt install ros-humble-ros-gz
sudo apt install ros-humble-ros-gz-bridge
```

### 2. Test ROS 2 Integration
```bash
source /opt/ros/humble/setup.bash
source /usr/share/gz/setup.sh
ros2 run ros_gz_bridge parameter_bridge
```

## Verification and Testing

### 1. Test Basic Functionality
```bash
gz sim --help
gz sim -r -v 1 empty.sdf
```

### 2. Test GUI Components
- Launch Gazebo GUI
- Verify 3D rendering works
- Test camera controls
- Verify physics simulation

### 3. Test with Sample Models
```bash
gz sim -r -v 1 -f https://fuel.gazebosim.org/1.0/openrobotics/models/Construction%20Block%20Big
```

## Uninstallation

To completely remove Gazebo:
```bash
sudo apt remove gz-garden
sudo apt autoremove
sudo rm /etc/apt/sources.list.d/gazebo-stable.list
```

## Post-Installation Recommendations

### 1. Download Useful Models
Visit https://fuel.gazebosim.org to download models for your projects.

### 2. Install Additional Tools
```bash
sudo apt install ros-humble-ros-gz-plugins
sudo apt install ros-humble-ros-gz-image
```

### 3. Documentation
Access local documentation:
```bash
man gz
```

## Common Simulation Assets

### 1. Popular Models
- Robots: TurtleBot3, PR2, Fetch, etc.
- Sensors: Cameras, LIDAR, IMU, GPS
- Environments: Rooms, outdoor spaces, objects

### 2. World Files
- Empty worlds for custom environments
- Pre-built environments (maze, office, etc.)
- Physics test environments

## Performance Optimization

### 1. Graphics Settings
- Use dedicated GPU
- Update graphics drivers
- Adjust rendering quality in GUI

### 2. Physics Settings
- Adjust real-time factor in world files
- Optimize collision meshes
- Use appropriate physics parameters

### 3. Resource Management
- Monitor system resources during simulation
- Close unnecessary applications
- Consider headless simulation for automated testing

## Summary

Gazebo Garden installation is now complete. You have:
- Installed the latest Gazebo simulation environment
- Configured your environment for Gazebo
- Set up a workspace for custom models and worlds
- Verified the installation with basic tests
- Prepared for ROS 2 integration

You're now ready to proceed with the digital twin and simulation modules in this book.

---

*Next: [Unity Installation Guide](unity.md)*