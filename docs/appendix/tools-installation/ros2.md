# Installation Guide: ROS 2 (Robot Operating System 2)

## Overview

This guide provides detailed instructions for installing ROS 2 (Humble Hawksbill) on Ubuntu 22.04 LTS. ROS 2 is the foundation for all robotic development in this book.

## System Requirements

### Minimum Requirements
- **Operating System**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Architecture**: 64-bit (amd64)
- **RAM**: 8 GB (16 GB recommended)
- **Disk Space**: 5 GB free space
- **Processor**: Multi-core processor (Intel Core i5 or equivalent AMD)

### Recommended Requirements
- **Operating System**: Ubuntu 22.04 LTS (Jammy Jellyfish)
- **Architecture**: 64-bit (amd64)
- **RAM**: 16 GB or more
- **Disk Space**: 20 GB free space for development
- **Processor**: Multi-core processor (Intel Core i7 or equivalent AMD)

## Pre-Installation Steps

### 1. System Update
Update your system packages to ensure compatibility:

```bash
sudo apt update
sudo apt upgrade
```

### 2. Set Locale
Ensure your locale is set to support UTF-8:

```bash
locale  # Check if output contains "LANG=en_US.UTF-8"
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

### 3. Add ROS 2 GPG Key
Add the ROS 2 GPG key to your system:

```bash
sudo apt update && sudo apt install curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
```

### 4. Add ROS 2 Repository
Add the ROS 2 repository to your sources list:

```bash
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

## Installation Process

### 1. Update Package Lists
```bash
sudo apt update
```

### 2. Install ROS 2 Packages
Install the full desktop version which includes all common packages:

```bash
sudo apt install ros-humble-desktop
```

For a minimal installation, use:
```bash
sudo apt install ros-humble-ros-base
```

### 3. Install Additional Development Tools
```bash
sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers
sudo apt install ros-humble-gazebo-ros-pkgs
sudo apt install python3-colcon-common-extensions
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator
```

### 4. Initialize rosdep
```bash
sudo rosdep init
rosdep update
```

## Environment Setup

### 1. Source ROS 2 Environment
Source the ROS 2 setup script to use ROS 2 commands:

```bash
source /opt/ros/humble/setup.bash
```

### 2. Permanent Environment Setup
Add the ROS 2 setup script to your bashrc to automatically source it:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

For zsh users:
```bash
echo "source /opt/ros/humble/setup.zsh" >> ~/.zshrc
```

### 3. Verify Installation
Test the installation by running a simple ROS 2 command:

```bash
ros2 --help
```

You should see the ROS 2 command-line interface help.

## Development Environment Setup

### 1. Create a ROS 2 Workspace
Create a workspace for your custom packages:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

### 2. Build the Workspace
```bash
source /opt/ros/humble/setup.bash
colcon build
```

### 3. Source the Workspace
```bash
source install/setup.bash
```

## Additional Python Packages

Install essential Python packages for robotics development:

```bash
pip3 install numpy scipy matplotlib
pip3 install transforms3d
pip3 install opencv-python
pip3 install pyyaml
```

## Verification and Testing

### 1. Test Basic Communication
Open two terminal windows and run:

Terminal 1:
```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_cpp talker
```

Terminal 2:
```bash
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_py listener
```

You should see messages being transmitted from the talker to the listener.

### 2. Test Parameter Server
```bash
source /opt/ros/humble/setup.bash
ros2 param list
```

### 3. Test Launch System
```bash
source /opt/ros/humble/setup.bash
ros2 launch --help
```

## Troubleshooting Common Issues

### 1. Package Not Found
If you encounter "package not found" errors:
```bash
sudo apt update
sudo apt upgrade
```

### 2. Permission Issues
If you have permission issues with serial devices:
```bash
sudo usermod -a -G dialout $USER
```
Log out and log back in for changes to take effect.

### 3. Missing Dependencies
For missing dependencies:
```bash
rosdep install --from-paths src --ignore-src -r -y
```

### 4. Python Package Issues
If Python packages don't work, try:
```bash
pip3 install --user package_name
```

## Advanced Configuration

### 1. Custom ROS Domain ID
To avoid interference in multi-robot environments:
```bash
export ROS_DOMAIN_ID=42  # Choose any number 0-255
```

### 2. DDS Middleware Configuration
For specific DDS middleware:
```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### 3. Network Configuration
For multi-machine communication, ensure:
- Same ROS_DOMAIN_ID on all machines
- Network connectivity between machines
- Firewall rules allow ROS 2 traffic (typically UDP ports 7400+)

## Uninstallation

To completely remove ROS 2:
```bash
sudo apt remove ros-humble-*
sudo apt autoremove
```

Remove the repository entry:
```bash
sudo rm /etc/apt/sources.list.d/ros2.list
```

## Post-Installation Recommendations

### 1. IDE Setup
Install VS Code with ROS extension:
```bash
sudo snap install code --classic
code --install-extension ms-iot.vscode-ros
```

### 2. Additional Tools
```bash
sudo apt install ros-humble-rqt ros-humble-rqt-common-plugins
sudo apt install ros-humble-turtlesim
```

### 3. Documentation
Access local documentation:
```bash
ros2 run rqt_doc rqt_doc
```

## Summary

ROS 2 installation is now complete. You have:
- Installed the ROS 2 Humble Hawksbill distribution
- Configured your environment
- Created a development workspace
- Verified the installation with basic tests

You're now ready to proceed with the ROS 2 modules in this book.

---

*Next: [Gazebo Installation Guide](gazebo.md)*