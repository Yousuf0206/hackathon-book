# Appendix: Tools Installation Guide

## Overview

This guide provides step-by-step instructions for installing all the software tools required for the Physical AI & Humanoid Robotics book. Each section corresponds to a specific technology stack used throughout the modules.

## Prerequisites

Before beginning the installation process, ensure your system meets the following requirements:

### System Requirements
- **Operating System**: Ubuntu 22.04 LTS (recommended) or Windows 11 with WSL2
- **RAM**: Minimum 16 GB (32 GB recommended for simulation)
- **Storage**: 50 GB free space minimum
- **Internet Connection**: Stable broadband for package downloads

### Windows Users: WSL2 Setup
For Windows users, WSL2 provides the best compatibility for ROS 2 and related tools:

1. Enable WSL2:
   ```bash
   wsl --install
   ```

2. Set Ubuntu 22.04 as default:
   ```bash
   wsl --set-default Ubuntu-22.04
   ```

3. Configure WSL2 for optimal performance:
   - Add to `.wslconfig` file in your Windows home directory:
   ```
   [wsl2]
   memory=16GB
   processors=8
   ```

## ROS 2 Installation (Humble Hawksbill)

### Ubuntu Installation
1. Set locale:
   ```bash
   locale  # check for UTF-8
   sudo apt update && sudo apt install locales
   sudo locale-gen en_US en_US.UTF-8
   sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
   export LANG=en_US.UTF-8
   ```

2. Add ROS 2 apt repository:
   ```bash
   sudo apt update && sudo apt install curl gnupg lsb-release
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   ```

3. Install ROS 2 packages:
   ```bash
   sudo apt update
   sudo apt install ros-humble-desktop
   sudo apt install python3-colcon-common-extensions
   sudo apt install python3-rosdep
   ```

4. Initialize rosdep:
   ```bash
   sudo rosdep init
   rosdep update
   ```

5. Source ROS 2 environment:
   ```bash
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

### Windows (WSL2) Installation
The same process applies in WSL2 Ubuntu.

## Gazebo Installation

### Gazebo Garden Installation
1. Add Gazebo repository:
   ```bash
   sudo curl -sSL http://get.gazebosim.org | sh
   ```

2. Install Gazebo Garden:
   ```bash
   sudo apt install gz-garden
   ```

3. Verify installation:
   ```bash
   gz sim
   ```

### Gazebo Classic (if needed for legacy compatibility)
1. Add repository:
   ```bash
   sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'
   wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -
   ```

2. Install:
   ```bash
   sudo apt update
   sudo apt install gazebo libgazebo-dev
   ```

## Python Development Environment

### Virtual Environment Setup
1. Install Python tools:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv
   ```

2. Create virtual environment:
   ```bash
   python3 -m venv ~/ros2_env
   source ~/ros2_env/bin/activate
   pip install --upgrade pip
   ```

### Essential Python Packages
```bash
pip install numpy scipy matplotlib
pip install opencv-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers openai
pip install speechrecognition pyaudio
pip install spacy
python -m spacy download en_core_web_sm
```

## NVIDIA Isaac Installation

### Prerequisites
1. Install NVIDIA drivers (minimum version 535):
   ```bash
   sudo apt install nvidia-driver-535
   sudo reboot
   ```

2. Install CUDA Toolkit:
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
   sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
   echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee /etc/apt/sources.list.d/cuda.list
   sudo apt update
   sudo apt install cuda-toolkit-12-3
   ```

### Isaac Sim Installation
1. Download Isaac Sim from NVIDIA Developer website
2. Extract and run the installer:
   ```bash
   tar -xf isaac_sim-4.0.0.tar.gz
   cd isaac_sim-4.0.0
   bash install.sh
   ```

3. Source the environment:
   ```bash
   source setup_omniverse_env.sh
   ```

### Isaac ROS Installation
1. Create a new ROS 2 workspace:
   ```bash
   mkdir -p ~/isaac_ros_ws/src
   cd ~/isaac_ros_ws
   ```

2. Clone Isaac ROS packages:
   ```bash
   git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git src/isaac_ros_common
   git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git src/isaac_ros_visual_slam
   git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_apriltag.git src/isaac_ros_apriltag
   git clone -b ros2 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_detectnet.git src/isaac_ros_detectnet
   ```

3. Build the workspace:
   ```bash
   source /opt/ros/humble/setup.bash
   colcon build --symlink-install
   source install/setup.bash
   ```

## Unity Installation (Optional)

### Unity Hub Installation
1. Download Unity Hub from Unity website
2. Install Unity Hub:
   ```bash
   # For Ubuntu
   sudo snap install unity-hub
   ```

3. Launch Unity Hub and install Unity 2022.3 LTS
4. Install required packages:
   - Universal Render Pipeline (URP)
   - Unity Recorder
   - Cinemachine

### ROS# Integration
1. Install ROS# package in Unity:
   - Add package via Unity Package Manager
   - URL: https://github.com/siemens/ros-sharp.git

## Development Tools

### IDE Setup
1. **Visual Studio Code**:
   ```bash
   sudo snap install code --classic
   ```

2. **Recommended Extensions**:
   - ROS
   - Python
   - C/C++
   - Docker
   - GitLens

### Additional Tools
```bash
sudo apt install git git-lfs
sudo apt install cmake build-essential
sudo apt install python3-dev python3-pip
sudo apt install ros-dev-tools
```

## Docker Setup (Optional but Recommended)

### Install Docker
```bash
sudo apt update
sudo apt install ca-certificates curl gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER
```

### Docker Compose for ROS 2
```bash
sudo apt install docker-compose-v2
```

## Verification Steps

### ROS 2 Verification
```bash
# Source environment
source /opt/ros/humble/setup.bash

# Test basic functionality
ros2 run demo_nodes_cpp talker &
ros2 run demo_nodes_py listener
```

### Python Packages Verification
```bash
python3 -c "import rclpy; print('ROS 2 Python bindings OK')"
python3 -c "import cv2; print('OpenCV OK')"
python3 -c "import torch; print('PyTorch OK')"
```

### Gazebo Verification
```bash
gz sim -v 4
```

## Troubleshooting Common Issues

### ROS 2 Installation Issues
- **Permission denied errors**: Check user permissions and add to dialout group:
  ```bash
  sudo usermod -a -G dialout $USER
  ```
- **Package not found**: Ensure correct Ubuntu codename in ROS repository setup

### GPU/CUDA Issues
- **CUDA not detected**: Verify NVIDIA driver installation:
  ```bash
  nvidia-smi
  nvidia-ml-py3
  ```
- **CUDA version mismatch**: Ensure PyTorch is installed with correct CUDA version

### Network Configuration
- **ROS 2 communication issues**: Check ROS_DOMAIN_ID and network configuration
- **Multi-robot communication**: Configure appropriate network settings

## Post-Installation Configuration

### Environment Setup
Add to `~/.bashrc`:
```bash
# ROS 2
source /opt/ros/humble/setup.bash

# Custom workspace
source ~/ros2_env/bin/activate
source ~/isaac_ros_ws/install/setup.bash

# Gazebo
source /usr/share/gazebo/setup.sh

# CUDA (if installed)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Workspace Initialization
```bash
mkdir -p ~/physical_ai_ws/src
cd ~/physical_ai_ws
colcon build
source install/setup.bash
```


*Next: [Lab Requirements](../lab-requirements/README.md)