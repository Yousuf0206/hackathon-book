# Chapter 3: Building ROS 2 Packages with rclpy

## Learning Objectives

After completing this chapter, you will be able to:
- Create and structure ROS 2 packages using Python
- Implement nodes with the rclpy client library
- Design and build complex node architectures
- Package and distribute ROS 2 applications
- Follow best practices for Python-based ROS 2 development

## Introduction to rclpy

rclpy is the Python client library for ROS 2, providing a Python API to the ROS 2 middleware. It allows Python developers to create ROS 2 nodes, publishers, subscribers, services, and actions with a familiar Python interface.

### Why Use rclpy?

- **Python Ecosystem**: Leverage Python's rich ecosystem of libraries for robotics
- **Rapid Prototyping**: Faster development cycles for algorithm prototyping
- **Scientific Computing**: Integration with NumPy, SciPy, and other scientific libraries
- **Machine Learning**: Easy integration with TensorFlow, PyTorch, and scikit-learn
- **Accessibility**: Lower barrier to entry for new roboticists

## Creating Your First ROS 2 Package

### Package Structure

A standard ROS 2 package follows this structure:

```
my_robot_package/
├── CMakeLists.txt          # Build configuration for C++
├── package.xml             # Package metadata
├── setup.cfg               # Python installation configuration
├── setup.py                # Python package setup
├── resource/               # Package resource files
├── test/                   # Unit tests
└── my_robot_package/       # Python module
    ├── __init__.py
    └── nodes/              # Python node implementations
        ├── __init__.py
        └── example_node.py
```

### Creating a Package with colcon

Use the `ros2 pkg create` command to create a new package:

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python my_robot_controller
```

This creates the basic package structure with the necessary configuration files.

### Package Configuration Files

#### package.xml

The `package.xml` file contains metadata about your package:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>my_robot_controller</name>
  <version>0.0.0</version>
  <description>Package for controlling humanoid robots</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <depend>rclpy</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>

  <test_depend>ament_copyright</test_depend>
  <test_depend>ament_flake8</test_depend>
  <test_depend>ament_pep257</test_depend>
  <test_depend>python3-pytest</test_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
```

#### setup.py

The `setup.py` file configures Python package installation:

```python
from setuptools import setup
from glob import glob
import os

package_name = 'my_robot_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='user@example.com',
    description='Package for controlling humanoid robots',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller = my_robot_controller.robot_controller:main',
            'sensor_processor = my_robot_controller.sensor_processor:main',
        ],
    },
)
```

## Advanced Node Implementation

### Node with Multiple Publishers and Subscribers

Here's an example of a more complex node that handles multiple communication patterns:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from my_robot_controller.srv import SetJointPosition
import numpy as np


class AdvancedRobotController(Node):
    def __init__(self):
        super().__init__('advanced_robot_controller')

        # Publishers
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Subscribers
        self.joystick_sub = self.create_subscription(
            Twist, 'joystick_cmd', self.joystick_callback, 10)
        self.imu_sub = self.create_subscription(
            String, 'imu_data', self.imu_callback, 10)

        # Service server
        self.service = self.create_service(
            SetJointPosition, 'set_joint_position', self.set_joint_position_callback)

        # Timer for periodic tasks
        self.timer = self.create_timer(0.1, self.control_loop)

        # Internal state
        self.joint_positions = {}
        self.target_positions = {}
        self.robot_state = 'idle'

        self.get_logger().info('Advanced Robot Controller initialized')

    def joystick_callback(self, msg):
        """Handle joystick commands"""
        self.cmd_vel_pub.publish(msg)
        self.get_logger().info(f'Received joystick command: linear={msg.linear.x}, angular={msg.angular.z}')

    def imu_callback(self, msg):
        """Handle IMU data"""
        # Process IMU data for balance control
        self.get_logger().info(f'IMU data: {msg.data}')

    def set_joint_position_callback(self, request, response):
        """Handle joint position service requests"""
        joint_name = request.joint_name
        position = request.position

        if joint_name in self.joint_positions:
            self.target_positions[joint_name] = position
            response.success = True
            response.message = f'Joint {joint_name} target set to {position}'
            self.get_logger().info(f'Set {joint_name} to {position}')
        else:
            response.success = False
            response.message = f'Joint {joint_name} not found'

        return response

    def control_loop(self):
        """Main control loop"""
        # Update joint states
        joint_msg = JointState()
        joint_msg.name = list(self.joint_positions.keys())
        joint_msg.position = list(self.joint_positions.values())
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.header.frame_id = 'base_link'

        self.joint_pub.publish(joint_msg)


def main(args=None):
    rclpy.init(args=args)
    node = AdvancedRobotController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

### Custom Message and Service Definitions

#### Creating Custom Messages

Create a `msg` directory in your package and define custom messages:

`msg/RobotCommand.msg`:
```
string command_type
float64[] joint_positions
geometry_msgs/Twist velocity
bool emergency_stop
```

`msg/RobotStatus.msg`:
```
string state
float64[] joint_angles
float64[] joint_velocities
bool is_balancing
bool is_moving
```

#### Creating Custom Services

Create a `srv` directory and define custom services:

`srv/SetJointPosition.srv`:
```
string joint_name
float64 position
---
bool success
string message
```

`srv/ExecuteTrajectory.srv`:
```
string trajectory_name
float64[] joint_positions
duration[] time_from_start
---
bool success
string message
```

## Package Dependencies and Build System

### Managing Dependencies

List all dependencies in `package.xml`:

```xml
<depend>rclpy</depend>
<depend>std_msgs</depend>
<depend>sensor_msgs</depend>
<depend>geometry_msgs</depend>
<depend>nav_msgs</depend>
<depend>tf2_ros</depend>
<depend>message_filters</depend>
<depend>cv_bridge</depend>
<depend>python3-numpy</depend>
<depend>python3-scipy</depend>
```

### Console Scripts

Define console scripts in `setup.py` to create executable commands:

```python
entry_points={
    'console_scripts': [
        'robot_controller = my_robot_controller.robot_controller:main',
        'sensor_processor = my_robot_controller.sensor_processor:main',
        'trajectory_executor = my_robot_controller.trajectory_executor:main',
        'calibration_tool = my_robot_controller.calibration_tool:main',
    ],
},
```

## Advanced rclpy Features

### Time and Timers

```python
from rclpy.time import Time
from rclpy.duration import Duration

class TimeAwareNode(Node):
    def __init__(self):
        super().__init__('time_aware_node')

        # Create timer with specific period
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20 Hz

        # Get current time
        self.start_time = self.get_clock().now()

    def timer_callback(self):
        current_time = self.get_clock().now()
        elapsed = current_time - self.start_time
        self.get_logger().info(f'Elapsed time: {elapsed.nanoseconds / 1e9:.3f}s')
```

### Parameter Callbacks

```python
class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with callbacks
        self.declare_parameter('control_frequency', 100)
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('robot_name', 'humanoid_robot')

        # Set parameter callback
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'control_frequency':
                # Adjust timer frequency based on parameter
                self.timer.timer_period_ns = int(1.0 / param.value * 1e9)
        return SetParametersResult(successful=True)
```

### TF Transformations

For humanoid robots, TF (Transform) is crucial for coordinate frame management:

```python
import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class TransformNode(Node):
    def __init__(self):
        super().__init__('transform_node')
        self.tf_broadcaster = TransformBroadcaster(self)

    def broadcast_transform(self, parent_frame, child_frame, translation, rotation):
        t = TransformStamped()

        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = parent_frame
        t.child_frame_id = child_frame

        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]

        t.transform.rotation.x = rotation[0]
        t.transform.rotation.y = rotation[1]
        t.transform.rotation.z = rotation[2]
        t.transform.rotation.w = rotation[3]

        self.tf_broadcaster.sendTransform(t)
```

## Testing and Quality Assurance

### Unit Testing with pytest

Create tests in the `test/` directory:

`test/test_nodes.py`:
```python
import unittest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from my_robot_controller.robot_controller import AdvancedRobotController


class TestRobotController(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.node = AdvancedRobotController()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

    def tearDown(self):
        self.node.destroy_node()
        rclpy.shutdown()

    def test_node_initialization(self):
        self.assertIsNotNone(self.node)
        self.assertEqual(self.node.robot_state, 'idle')

    def test_service_response(self):
        # Test service call
        future = self.node.set_joint_position('test_joint', 1.0)
        self.assertIsNotNone(future)


if __name__ == '__main__':
    unittest.main()
```

### Linting and Code Quality

Use tools like flake8 and pylint for code quality:

```bash
# Install linters
pip install flake8 pylint

# Run linters
ament_flake8 my_robot_controller
```

## Package Distribution

### Creating Launch Files

Launch files allow you to start multiple nodes with a single command:

`launch/robot_control.launch.py`:
```python
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_robot_controller',
            executable='robot_controller',
            name='robot_controller',
            parameters=[
                os.path.join(get_package_share_directory('my_robot_controller'), 'config', 'robot_params.yaml')
            ],
            output='screen'
        ),
        Node(
            package='my_robot_controller',
            executable='sensor_processor',
            name='sensor_processor',
            output='screen'
        )
    ])
```

### Configuration Files

Store configuration in YAML files:

`config/robot_params.yaml`:
```yaml
robot_controller:
  ros__parameters:
    control_frequency: 100
    max_velocity: 1.0
    robot_name: "humanoid_robot"
    joint_limits:
      hip_joint: [-1.57, 1.57]
      knee_joint: [0.0, 2.0]
      ankle_joint: [-0.78, 0.78]
```

## Best Practices for rclpy Development

### 1. Error Handling
```python
def safe_divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        self.get_logger().error(f'Division by zero: {a} / {b}')
        return 0.0
```

### 2. Resource Management
```python
def destroy_node(self):
    # Clean up resources
    if hasattr(self, 'timer'):
        self.timer.cancel()
    super().destroy_node()
```

### 3. Logging
```python
def some_method(self):
    self.get_logger().debug('Detailed debug information')
    self.get_logger().info('Normal operation info')
    self.get_logger().warn('Warning message')
    self.get_logger().error('Error occurred')
    self.get_logger().fatal('Fatal error')
```

### 4. Performance Considerations
- Use appropriate QoS profiles for different data types
- Minimize message copying in callbacks
- Use efficient data structures (NumPy arrays for numerical data)

## Summary

This chapter covered the fundamentals of creating ROS 2 packages with rclpy, including package structure, node implementation, custom messages and services, and best practices. These concepts are essential for building robust humanoid robot control systems.

## Exercises

1. Create a ROS 2 package for a simple humanoid robot with joint control
2. Implement a service that calculates inverse kinematics for an arm
3. Design a launch file that starts multiple nodes for robot control
4. Create custom messages for humanoid robot-specific data types
5. Implement parameter validation and callbacks

## Further Reading

- ROS 2 Python Developer Guide: https://docs.ros.org/en/humble/How-To-Guides/Developing-Python-package.html
- rclpy API Documentation: https://docs.ros2.org/latest/api/rclpy/
- Ament Python: https://github.com/ament/ament_python

---

*Next: [Chapter 4: Robot Description Formats (URDF for humanoid robots)](../chapter4-urdf/README.md)*