# Appendix: Hardware Setup for Physical AI & Humanoid Robotics

## Overview

This appendix provides detailed hardware requirements and setup recommendations for implementing the concepts and examples in this book. The hardware requirements vary depending on the specific applications and scale of implementation.

## Minimum Hardware Requirements

### Development Station
- **CPU**: Intel Core i7-10700K or AMD Ryzen 7 3700X (8 cores/16 threads)
- **RAM**: 16 GB DDR4 (32 GB recommended for complex simulations)
- **GPU**: NVIDIA RTX 3060 or equivalent with minimum 12 GB VRAM
- **Storage**: 500 GB SSD (1 TB recommended for simulation assets)
- **OS**: Ubuntu 22.04 LTS or Windows 11 with WSL2
- **Network**: Gigabit Ethernet for robot communication

### Network Infrastructure
- **Router**: Gigabit capable with Quality of Service (QoS) settings
- **Bandwidth**: Minimum 100 Mbps dedicated for robotics applications
- **Latency**: &lt;10ms for real-time control applications

## Recommended Hardware for Humanoid Robotics

### Computing Units
- **On-board Computer**: NVIDIA Jetson Orin AGX (for humanoid robots)
  - 8-core ARM CPU
  - 2048-core NVIDIA Ampere GPU
  - 32 GB LPDDR5 memory
  - 128 GB eMMC storage

- **Alternative**: Raspberry Pi 4 with Coral USB Accelerator
  - 4-core ARM Cortex-A72
  - 4 GB LPDDR4
  - Coral Edge TPU for AI acceleration

### Sensors
- **Cameras**:
  - Stereo Vision: ZED 2i or Intel RealSense D455
  - RGB: 1080p minimum, 4K preferred
  - FOV: 90-120 degrees
  - Frame Rate: 30-60 FPS

- **IMU (Inertial Measurement Unit)**:
  - Bosch BNO055 or ADIS16470
  - 9-axis (accelerometer, gyroscope, magnetometer)
  - High accuracy for balance control

- **LIDAR**:
  - 2D: Hokuyo URG-04LX-UG01 or RPLIDAR A3
  - 3D: Ouster OS1-64 or Velodyne VLP-16 (for advanced perception)

- **Force/Torque Sensors**:
  - ATI Gamma or F/T Sensors
  - For precise manipulation and balance feedback

### Actuators for Humanoid Robots
- **Servo Motors**: Dynamixel X-series or Herkulex DRS-0101
  - High torque-to-weight ratio
  - Position, velocity, and current feedback
  - Serial communication capability

- **Hydraulic/Pneumatic Systems**: For high-power applications
  - Custom systems for full-size humanoid robots
  - Compressor and control valves

## Specialized Hardware for Different Applications

### ROS 2 Development (Module 1)
- **Additional Requirements**:
  - Real-time kernel patches for Ubuntu (PREEMPT_RT)
  - CAN bus interface for hardware communication
  - USB 3.0 ports for multiple sensors
  - Multi-serial interface for multiple devices

### Simulation and Digital Twins (Module 2)
- **Enhanced GPU Requirements**:
  - NVIDIA RTX 4080 or RTX 6000 Ada Generation (48GB) for high-fidelity simulation
  - Multiple GPU support for parallel simulation
  - Ray tracing capabilities for realistic rendering

- **Storage Considerations**:
  - Fast NVMe SSD for simulation assets
  - Large storage capacity (2-4 TB) for simulation data

### AI Perception Systems (Module 3)
- **NVIDIA Isaac Compatible Hardware**:
  - Jetson AGX Orin Development Kit
  - Isaac ROS compatible sensors and cameras
  - High-speed interfaces (MIPI CSI-2, GMSL2)

- **Training Hardware** (if developing custom models):
  - NVIDIA RTX 4090 or A6000 for model training
  - Multiple GPUs for distributed training

### Vision-Language-Action Systems (Module 4)
- **Audio Processing**:
  - USB microphone array (ReSpeaker 6-Mic Array)
  - Audio interface with ASIO support
  - Noise cancellation capabilities

- **Edge AI Acceleration**:
  - NVIDIA Jetson Orin for on-robot processing
  - Coral Edge TPU for lightweight inference
  - Intel Neural Compute Stick 2 (alternative)

## Hardware Integration Guidelines

### Power Management
- **Power Distribution**: Centralized power distribution with individual fuses
- **Battery Systems**:
  - LiPo batteries for mobility (22.2V, 10000mAh minimum)
  - Battery Management System (BMS) for safety
  - Power monitoring and low-battery alerts

### Communication Interfaces
- **Ethernet**: Hardwired connections for critical control systems
- **WiFi 6**: For non-critical data transmission
- **Bluetooth 5.0+**: For short-range communication
- **CAN Bus**: For real-time, deterministic communication

### Safety Considerations
- **Emergency Stop**: Hardware emergency stop button with direct power cutoff
- **Safety Sensors**: Collision detection and prevention systems
- **Enclosure**: Proper IP rating (IP54 minimum) for environmental protection
- **Cooling**: Adequate cooling for high-performance computing components

## Budget Considerations

### Educational Setup (Low Budget)
- Development station: $1,500-3,000
- Basic sensors: $500-1,000
- Entry-level actuators: $1,000-2,000
- **Total**: $3,000-6,000

### Professional Setup (Medium Budget)
- Development station: $3,000-6,000
- Professional sensors: $2,000-5,000
- Quality actuators: $3,000-8,000
- Simulation hardware: $2,000-4,000
- **Total**: $10,000-23,000

### Research Setup (High Budget)
- High-end development station: $6,000-15,000
- Research-grade sensors: $10,000-25,000
- Professional actuators: $15,000-40,000
- Specialized equipment: $10,000-30,000
- **Total**: $41,000-110,000

## Vendor Recommendations

### Robotics Platforms
- **Trossen Robotics**: Humanoid robot kits and components
- **Robotis**: Dynamixel servos and OpenMANIPULATOR
- **Unitree**: Quadruped and humanoid platforms
- **Boston Dynamics**: Advanced platforms (research license)

### Electronics Suppliers
- **Digi-Key**: Electronic components and sensors
- **Mouser Electronics**: Specialty robotics components
- **SparkFun**: Prototyping and development boards
- **Adafruit**: Educational and maker-friendly components

### Specialized Robotics Components
- **Herkulex**: High-performance servo systems
- **Dynamixel**: Premium servo motors
- **ATI Industrial Automation**: Force/torque sensors
- **Hokuyo**: LIDAR systems

## Assembly and Integration Tips

### Mechanical Assembly
1. Follow manufacturer's torque specifications
2. Use thread locker on critical fasteners
3. Implement proper cable management
4. Allow for thermal expansion and contraction

### Electrical Integration
1. Separate power and signal grounds where possible
2. Use proper EMI/RFI shielding
3. Implement power filtering for sensitive components
4. Test individual subsystems before integration

### Software Integration
1. Verify communication protocols before mechanical integration
2. Test sensor calibration procedures
3. Validate safety systems before full operation
4. Implement logging for debugging and analysis

---

*Next: [Tools Installation Guide](../tools-installation/README.md)*