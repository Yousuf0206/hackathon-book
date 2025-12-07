# Installation Guide: Unity (Unity Hub and Unity 2022.3 LTS)

## Overview

This guide provides detailed instructions for installing Unity Hub and Unity 2022.3 LTS, which is used for high-fidelity robot visualization and Unity-ROS integration in Module 2 of this book.

## System Requirements

### Minimum Requirements
- **Operating System**:
  - Windows 10/11 (64-bit)
  - Ubuntu 20.04 LTS or 22.04 LTS (Unity Hub only, limited Unity support)
  - macOS 10.15+ (for Mac users)
- **Architecture**: 64-bit (x86_64)
- **RAM**: 8 GB (16 GB recommended)
- **GPU**: Graphics card with DX10+ support and 1GB+ VRAM
- **Disk Space**: 20 GB free space for Unity Hub + Unity Editor
- **Processor**: Intel Core i5 or AMD equivalent with 4+ cores

### Recommended Requirements
- **Operating System**: Windows 11 (64-bit) or Ubuntu 22.04 LTS
- **Architecture**: 64-bit (x86_64)
- **RAM**: 32 GB or more for complex scenes
- **GPU**: NVIDIA RTX 3060 or AMD equivalent with 8GB+ VRAM
- **Disk Space**: 50 GB free space for development
- **Processor**: Intel Core i7 or AMD Ryzen 7 with 8+ cores

## Pre-Installation Steps

### 1. System Preparation (Windows)
Ensure your system is up to date:
- Update Windows to the latest version
- Install latest Visual Studio redistributables
- Ensure .NET Framework 4.8 is installed

### 2. Check Hardware Compatibility
Verify your GPU supports Unity's rendering requirements:
```bash
# On Windows, check DirectX version:
dxdiag
```

### 3. Create Unity Account
Visit https://id.unity.com and create a Unity ID account, which is required for downloading Unity.

## Installation Process

### 1. Install Unity Hub
Unity Hub is the recommended way to manage Unity installations:

**Windows Installation:**
1. Download Unity Hub from https://unity.com/download
2. Run the installer as Administrator
3. Follow the installation wizard
4. Launch Unity Hub after installation completes

**Ubuntu Installation:**
```bash
# Install Unity Hub via Snap (recommended):
sudo snap install unityhub

# Or download the .AppImage from Unity website and make executable:
chmod +x UnityHub.AppImage
./UnityHub.AppImage
```

### 2. Sign In to Unity Hub
1. Open Unity Hub
2. Click "Sign In" in the top-right corner
3. Use your Unity ID credentials
4. Verify your account if necessary

### 3. Install Unity 2022.3 LTS
1. In Unity Hub, go to the "Installs" tab
2. Click "Add" to add a new Unity version
3. Select "2022.3.22f1" (or the latest 2022.3 LTS version)
4. In the installer, select the following modules:
   - **Unity Editor**: Standard or Pro (based on your license)
   - **Android Build Support** (if targeting Android)
   - **Windows Build Support** (IL2CPP)
   - **Visual Studio Tools for Unity**
   - **Unity Package Manager**
   - **Documentation**

### 4. Install Additional Packages
After Unity installation completes:
1. Open Unity Hub
2. Go to the "Modules" tab
3. Install additional modules as needed:
   - **Unity Recorder** (for recording simulation footage)
   - **Cinemachine** (for camera systems)
   - **ProBuilder** (for quick prototyping)
   - **ProGrids** (for grid-based editing)

## Environment Setup

### 1. Configure Unity Hub
1. In Unity Hub preferences:
   - Set default project location
   - Configure external editor (Visual Studio, VS Code, etc.)
   - Set up proxy if behind corporate firewall

### 2. Install Visual Studio Community
Unity works best with Visual Studio for scripting:
1. Download Visual Studio Community from Microsoft
2. During installation, select:
   - Game development with Unity workload
   - .NET desktop development
   - C++ development tools

### 3. Configure Unity Editor Settings
After first launch of Unity:
1. Go to Edit → Preferences (Windows) or Unity → Preferences (Mac)
2. Configure:
   - External Tools → External Script Editor (set to your preferred editor)
   - General → Set up appropriate color space (Linear recommended)
   - Player → Configure target platform settings

## ROS# Integration Setup

### 1. Install ROS# Package
1. Open Unity and create a new 3D project
2. Go to Window → Package Manager
3. Click the "+" button and select "Add package from git URL..."
4. Enter: `https://github.com/siemens/ros-sharp.git`
5. Install the following packages:
   - ROS Bridge Client
   - ROS Bridge Library
   - Unity Companion Packages

### 2. Configure ROS Communication
1. Add ROS communication components to your scene
2. Configure ROS Bridge connection settings
3. Test connection to ROS master

## Verification and Testing

### 1. Test Basic Unity Functionality
1. Create a new 3D project in Unity Hub
2. Verify the editor opens correctly
3. Test basic scene creation:
   - Create a cube (GameObject → 3D Object → Cube)
   - Add lighting (GameObject → Light → Directional Light)
   - Add camera (GameObject → Camera)
   - Play the scene (Press Play button)

### 2. Test Scripting Environment
1. Create a new C# script (Assets → Create → C# Script)
2. Verify the external editor integration works
3. Test basic scripting functionality

### 3. Test Package Installation
1. Verify Unity Recorder package works
2. Test Cinemachine camera system
3. Confirm all required packages are properly installed

## Development Environment Setup

### 1. Create Unity Workspace
Create a dedicated workspace for robotics projects:
```bash
mkdir -p ~/unity_robotics_projects
# Or on Windows: C:\UnityRoboticsProjects
```

### 2. Set Up Project Templates
1. Create a robotics project template with:
   - Standard assets
   - ROS# integration
   - Camera and lighting setup
   - Physics configuration

### 3. Install Additional Useful Packages
In Unity Package Manager, install:
- **Universal Render Pipeline (URP)**: For modern rendering
- **Post Processing**: For visual effects
- **Terrain Tools**: For outdoor environments
- **TextMeshPro**: For better text rendering

## Troubleshooting Common Issues

### 1. Installation Failures
If Unity installation fails:
- Ensure sufficient disk space
- Run Unity Hub as Administrator
- Temporarily disable antivirus software
- Check internet connection stability

### 2. Graphics Issues
If you encounter rendering problems:
- Update graphics drivers
- Verify DirectX/OpenGL support
- Try running Unity in compatibility mode
- Check GPU memory requirements

### 3. Performance Issues
For better performance:
- Close unnecessary applications
- Increase virtual memory
- Use SSD storage for projects
- Adjust Unity Quality Settings (Edit → Project Settings → Quality)

### 4. ROS Integration Issues
For ROS# connection problems:
- Verify ROS master is running
- Check network connectivity
- Confirm ROS# configuration settings
- Review firewall settings

## Advanced Configuration

### 1. Physics Configuration
Configure physics for robotics simulation:
- Set appropriate gravity settings
- Configure collision detection
- Adjust physics update rate

### 2. Rendering Configuration
Optimize rendering for robotics applications:
- Set appropriate render pipeline (URP recommended)
- Configure lighting settings
- Adjust post-processing effects

### 3. Build Settings
Configure for robotics applications:
- Set appropriate target platform
- Configure compression settings
- Set up build automation

## Integration with Other Tools

### 1. ROS Bridge Setup
Configure ROS bridge for Unity-ROS communication:
1. Install rosbridge_suite in ROS:
   ```bash
   sudo apt install ros-humble-rosbridge-suite
   ```
2. Launch ROS bridge server
3. Configure Unity ROS# connection

### 2. Asset Integration
Import robotics-specific assets:
- Robot models (URDF/FBX format)
- Environment models
- Sensor visualization assets

## Uninstallation

### To Remove Unity Hub:
**Windows:**
- Use Windows "Add or Remove Programs"
- Or run the Unity Hub uninstaller

**Ubuntu:**
```bash
sudo snap remove unityhub
```

### To Remove Unity Editor Versions:
1. Open Unity Hub
2. Go to Installs tab
3. Click the gear icon next to the version
4. Select "Remove"

## Post-Installation Recommendations

### 1. Project Organization
Set up a consistent project structure:
```
UnityRoboticsProjects/
├── 01_RosIntegration/
├── 02_SensorSimulation/
├── 03_RobotControl/
└── AssetsLibrary/
```

### 2. Learning Resources
- Complete Unity's official tutorials
- Review Unity Robotics documentation
- Explore Unity ML-Agents for AI integration

### 3. Performance Monitoring
- Monitor frame rates during simulation
- Profile memory usage
- Optimize scenes for real-time performance

## Version Control Setup

### 1. Git Configuration for Unity
Unity projects require special Git configuration:
1. Install Git LFS:
   ```bash
   git lfs install
   ```
2. Use appropriate .gitignore for Unity projects
3. Consider using Plastic SCM for better Unity asset handling

### 2. Recommended Git Workflow
- Use feature branches for new functionality
- Regular commits with descriptive messages
- Backup projects to remote repositories

## Summary

Unity Hub and Unity 2022.3 LTS installation is now complete. You have:
- Installed Unity Hub for version management
- Configured Unity 2022.3 LTS with necessary modules
- Set up ROS# integration for robotics applications
- Verified the installation with basic tests
- Prepared for Unity-ROS integration projects

You're now ready to proceed with the Unity integration modules in this book.

---

*Next: [NVIDIA Isaac Installation Guide](isaac.md)*