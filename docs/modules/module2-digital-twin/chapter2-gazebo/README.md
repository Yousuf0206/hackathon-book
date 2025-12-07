# Chapter 2: Gazebo Fundamentals

## Learning Objectives

After completing this chapter, you will be able to:
- Set up and configure Gazebo simulation environments
- Create and import robot models into Gazebo
- Implement sensor configurations and physics properties
- Design simulation scenarios for humanoid robotics
- Integrate Gazebo with ROS 2 for closed-loop control

## Introduction to Gazebo

Gazebo is a 3D dynamic simulator that provides realistic physics simulation, high-quality graphics, and convenient programmatic interfaces. For humanoid robotics, Gazebo serves as a critical tool for testing algorithms, validating control systems, and training AI models in a safe, reproducible environment.

### Why Gazebo for Humanoid Robotics?

- **Realistic Physics**: Accurate simulation of rigid body dynamics, contacts, and collisions
- **Sensor Simulation**: Cameras, LIDAR, IMU, force/torque sensors, and more
- **Plugin Architecture**: Extensible with custom sensors and controllers
- **ROS Integration**: Seamless communication with ROS 2 systems
- **Visualization**: High-quality 3D rendering for debugging and monitoring

## Gazebo Architecture and Components

### Core Components

Gazebo consists of several key components:

1. **Physics Engine**: Handles rigid body dynamics (ODE, Bullet, SimBody)
2. **Sensor System**: Simulates various sensor types with realistic noise models
3. **Rendering Engine**: Provides 3D visualization (OGRE-based)
4. **Communication Layer**: Handles inter-process communication (ZeroMQ/Protobuf)
5. **Plugin Interface**: Extends functionality through plugins

### Simulation Environment Structure

```
Gazebo World
├── Physics Engine Configuration
├── Models (Robots, Objects, Environment)
├── Lights
├── Plugins (Custom functionality)
└── GUI (Visualization and Control)
```

## Setting Up Gazebo with ROS 2

### Installation and Configuration

```bash
# Install Gazebo Garden (recommended for ROS 2 Humble)
sudo apt install ros-humble-gazebo-*

# Install Gazebo ROS packages
sudo apt install ros-humble-gazebo-ros ros-humble-gazebo-ros-pkgs
```

### Basic Gazebo Launch

```xml
<!-- launch/gazebo.launch.py -->
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    world_file = PathJoinSubstitution([
        FindPackageShare('my_robot_gazebo'),
        'worlds',
        'humanoid_world.world'
    ])

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': world_file,
            'verbose': 'true'
        }.items()
    )

    return LaunchDescription([
        gazebo
    ])
```

## Creating Gazebo Worlds

### World File Structure

Gazebo worlds are defined using SDF (Simulation Description Format):

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="humanoid_world">
    <!-- Physics engine configuration -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <!-- GUI configuration -->
    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>-5 0 2 0 0.4 0</pose>
      </camera>
    </gui>

    <!-- Include models -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Custom models -->
    <model name="humanoid_robot">
      <!-- Robot definition would go here -->
    </model>

    <!-- Additional objects -->
    <model name="table">
      <pose>2 0 0 0 0 0</pose>
      <link name="table_link">
        <visual name="visual">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>1.0 0.8 0.8</size>
            </box>
          </geometry>
        </collision>
        <inertial>
          <mass>50.0</mass>
          <inertia>
            <ixx>1.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>1.0</iyy>
            <iyz>0.0</iyz>
            <izz>1.0</izz>
          </inertia>
        </inertial>
      </link>
    </model>
  </world>
</sdf>
```

### Physics Configuration for Humanoid Robots

For humanoid robots, careful physics configuration is essential:

```xml
<physics type="ode">
  <!-- Smaller step size for stability with humanoid dynamics -->
  <max_step_size>0.001</max_step_size>

  <!-- High update rate for real-time control -->
  <real_time_update_rate>1000</real_time_update_rate>

  <!-- Real-time factor of 1.0 for real-time simulation -->
  <real_time_factor>1.0</real_time_factor>

  <!-- Standard gravity -->
  <gravity>0 0 -9.8</gravity>

  <!-- ODE-specific parameters -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>20</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.00001</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

## Robot Integration in Gazebo

### URDF to SDF Conversion

Gazebo can directly load URDF files, but SDF provides more control:

```xml
<!-- Example of including a URDF robot in Gazebo -->
<model name="my_humanoid">
  <include>
    <uri>file://$(find my_robot_description)/urdf/my_humanoid.urdf</uri>
  </include>

  <!-- Additional Gazebo-specific configurations -->
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/my_humanoid</robotNamespace>
    <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
  </plugin>
</model>
```

### Gazebo-Specific Robot Configuration

For better integration, create a Gazebo-specific model:

```xml
<sdf version="1.7">
  <model name="humanoid_robot">
    <!-- Initial pose -->
    <pose>0 0 1.0 0 0 0</pose>

    <!-- Links with Gazebo-specific properties -->
    <link name="base_link">
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.1</iyy>
          <iyz>0.0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>

      <visual name="visual">
        <geometry>
          <box><size>0.3 0.3 0.3</size></box>
        </geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
          <specular>0.1 0.1 0.1 1</specular>
        </material>
      </visual>

      <collision name="collision">
        <geometry>
          <box><size>0.3 0.3 0.3</size></box>
        </geometry>
      </collision>

      <!-- Gazebo-specific properties -->
      <self_collide>false</self_collide>
      <kinematic>false</kinematic>
      <gravity>true</gravity>
    </link>

    <!-- Joint definition -->
    <joint name="hip_joint" type="revolute">
      <parent>base_link</parent>
      <child>left_thigh</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>100</effort>
          <velocity>3</velocity>
        </limit>
        <dynamics>
          <damping>1.0</damping>
          <friction>0.1</friction>
        </dynamics>
      </axis>
    </joint>
  </model>
</sdf>
```

## Sensor Integration

### Camera Sensors

```xml
<sensor name="camera" type="camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="head_camera">
    <horizontal_fov>1.3962634</horizontal_fov> <!-- 80 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <frame_name>camera_frame</frame_name>
    <topic_name>camera/image_raw</topic_name>
  </plugin>
</sensor>
```

### IMU Sensors

```xml
<sensor name="imu" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
    <topic_name>imu/data</topic_name>
    <body_name>torso</body_name>
    <frame_name>imu_frame</frame_name>
  </plugin>
</sensor>
```

### Force/Torque Sensors

```xml
<sensor name="ft_sensor" type="force_torque">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <force_torque>
    <frame>child</frame>
    <measure_direction>child_to_parent</measure_direction>
  </force_torque>
  <plugin name="ft_plugin" filename="libgazebo_ros_ft_sensor.so">
    <topic_name>wrench</topic_name>
    <joint_name>ankle_joint</joint_name>
  </plugin>
</sensor>
```

## Control Integration

### Gazebo ROS Control Plugin

The gazebo_ros_control plugin bridges Gazebo and ROS 2:

```xml
<gazebo>
  <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
    <robotNamespace>/humanoid</robotNamespace>
    <robotSimType>gazebo_ros_control/DefaultRobotHWSim</robotSimType>
    <controlPeriod>0.001</controlPeriod> <!-- 1kHz -->
  </plugin>
</gazebo>
```

### Joint Controller Configuration

Create a controller configuration file:

`config/controllers.yaml`:
```yaml
controller_manager:
  ros__parameters:
    update_rate: 1000  # Hz

    joint_state_controller:
      type: joint_state_controller/JointStateController

    left_leg_controller:
      type: position_controllers/JointGroupPositionController
      joints:
        - left_hip_joint
        - left_knee_joint
        - left_ankle_joint

    right_leg_controller:
      type: position_controllers/JointGroupPositionController
      joints:
        - right_hip_joint
        - right_knee_joint
        - right_ankle_joint

    left_arm_controller:
      type: position_controllers/JointGroupPositionController
      joints:
        - left_shoulder_joint
        - left_elbow_joint

    right_arm_controller:
      type: position_controllers/JointGroupPositionController
      joints:
        - right_shoulder_joint
        - right_elbow_joint
```

## Advanced Simulation Techniques

### Contact Sensors

For foot contact detection in humanoid walking:

```xml
<sensor name="left_foot_contact" type="contact">
  <always_on>true</always_on>
  <update_rate>1000</update_rate>
  <contact>
    <collision>left_foot_collision</collision>
  </contact>
  <plugin name="contact_plugin" filename="libgazebo_ros_bumper.so">
    <frame_name>left_foot</frame_name>
    <topic_name>left_foot_contact</topic_name>
  </plugin>
</sensor>
```

### Ray Sensors (LIDAR Simulation)

```xml
<sensor name="laser" type="ray">
  <always_on>true</always_on>
  <update_rate>40</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-1.570796</min_angle>  <!-- -90 degrees -->
        <max_angle>1.570796</max_angle>   <!-- 90 degrees -->
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <plugin name="laser_plugin" filename="libgazebo_ros_laser.so">
    <topic_name>scan</topic_name>
    <frame_name>laser_frame</frame_name>
  </plugin>
</sensor>
```

## Simulation Scenarios for Humanoid Robots

### Balance Testing Scenario

Create a scenario to test humanoid balance:

```xml
<!-- worlds/balance_test.world -->
<sdf version="1.7">
  <world name="balance_test">
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
      <gravity>0 0 -9.8</gravity>
    </physics>

    <include>
      <uri>model://ground_plane</uri>
    </include>

    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Platform that can be moved to test balance -->
    <model name="movable_platform">
      <pose>0 0 0 0 0 0</pose>
      <link name="platform_link">
        <collision name="collision">
          <geometry>
            <box><size>2.0 2.0 0.1</size></box>
          </geometry>
        </collision>
        <visual name="visual">
          <geometry>
            <box><size>2.0 2.0 0.1</size></box>
          </geometry>
          <material>
            <diffuse>0.5 0.5 0.5 1</diffuse>
          </material>
        </visual>
        <inertial>
          <mass>100.0</mass>
          <inertia>
            <ixx>10.0</ixx>
            <ixy>0.0</ixy>
            <ixz>0.0</ixz>
            <iyy>10.0</iyy>
            <iyz>0.0</iyz>
            <izz>20.0</izz>
          </inertia>
        </inertial>
      </link>

      <!-- Actuator to move the platform -->
      <plugin name="platform_actuator" filename="libgazebo_ros_p3d.so">
        <alwaysOn>true</alwaysOn>
        <updateRate>100</updateRate>
        <bodyName>platform_link</bodyName>
        <topicName>platform/pose</topicName>
        <gaussianNoise>0.01</gaussianNoise>
      </plugin>
    </model>

    <!-- Humanoid robot -->
    <model name="humanoid_robot" static="false">
      <!-- Robot definition -->
    </model>
  </world>
</sdf>
```

### Stair Climbing Scenario

```xml
<!-- worlds/stairs.world -->
<world name="stairs_world">
  <!-- ... physics configuration ... -->

  <!-- Stairs object -->
  <model name="stairs">
    <pose>2 0 0 0 0 0</pose>
    <link name="stairs_link">
      <!-- Create individual steps -->
      <visual name="step1_visual">
        <pose>0 0 0.1 0 0 0</pose>
        <geometry>
          <box><size>1.0 1.0 0.2</size></box>
        </geometry>
      </visual>
      <collision name="step1_collision">
        <pose>0 0 0.1 0 0 0</pose>
        <geometry>
          <box><size>1.0 1.0 0.2</size></box>
        </geometry>
      </collision>

      <visual name="step2_visual">
        <pose>0 0 0.3 0 0 0</pose>
        <geometry>
          <box><size>1.0 1.0 0.2</size></box>
        </geometry>
      </visual>
      <collision name="step2_collision">
        <pose>0 0 0.3 0 0 0</pose>
        <geometry>
          <box><size>1.0 1.0 0.2</size></box>
        </geometry>
      </collision>
      <!-- Additional steps ... -->
    </link>
  </model>
</world>
```

## Performance Optimization

### Efficient Simulation Settings

For humanoid robot simulation, balance accuracy and performance:

```xml
<physics type="ode">
  <!-- Use smaller step size for humanoid stability -->
  <max_step_size>0.001</max_step_size>

  <!-- Higher update rate for control -->
  <real_time_update_rate>1000</real_time_update_rate>

  <!-- Optimized ODE settings -->
  <ode>
    <solver>
      <type>quick</type>
      <iters>20</iters>  <!-- Balance between accuracy and speed -->
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.00001</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Collision Optimization

Use simplified collision geometry for performance:

```xml
<!-- For complex visual models, use simplified collision geometry -->
<link name="complex_visual_link">
  <!-- Detailed visual geometry -->
  <visual name="detailed_visual">
    <geometry>
      <mesh><uri>model://my_robot/meshes/complex_shape.dae</uri></mesh>
    </geometry>
  </visual>

  <!-- Simplified collision geometry -->
  <collision name="simple_collision">
    <geometry>
      <box><size>0.2 0.1 0.3</size></box>
    </geometry>
  </collision>

  <!-- Or use multiple simple shapes -->
  <collision name="collision_1">
    <geometry>
      <cylinder><radius>0.05</radius><length>0.2</length></cylinder>
    </geometry>
  </collision>
</link>
```

## Debugging and Visualization

### Model Debugging

Enable physics visualization in Gazebo:

```xml
<world name="debug_world">
  <!-- ... -->

  <gui>
    <camera name="debug_camera">
      <pose>3 3 2 0 0.5 2.356</pose>
    </camera>
    <plugin filename="gazebo_gui" name="gazebo_gui">
      <physics_visualization>true</physics_visualization>
    </plugin>
  </gui>
</world>
```

### Logging and Analysis

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
import csv

class SimulationAnalyzer(Node):
    def __init__(self):
        super().__init__('simulation_analyzer')

        # Subscribe to joint states
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)

        # Log data for analysis
        self.data_log = []

        # Timer for periodic analysis
        self.analysis_timer = self.create_timer(1.0, self.analysis_callback)

    def joint_callback(self, msg):
        # Log joint data
        data_point = {
            'timestamp': self.get_clock().now().nanoseconds,
            'positions': list(msg.position),
            'velocities': list(msg.velocity),
            'efforts': list(msg.effort)
        }
        self.data_log.append(data_point)

    def analysis_callback(self):
        # Perform analysis on logged data
        if len(self.data_log) > 100:  # Only analyze if enough data
            self.perform_stability_analysis()

    def perform_stability_analysis(self):
        # Analyze robot stability metrics
        # Calculate CoM, ZMP, etc.
        pass
```

## Best Practices for Humanoid Simulation

### 1. Model Accuracy vs. Performance
- Use realistic but not overly complex models
- Balance visual detail with collision simplicity
- Optimize for your specific use case

### 2. Physics Tuning
- Start with conservative parameters and tune gradually
- Match real-world robot dynamics as closely as possible
- Validate simulation results against real robot behavior

### 3. Sensor Configuration
- Add realistic noise models to sensor data
- Configure appropriate update rates
- Include sensor limitations (FOV, range, etc.)

### 4. Safety and Validation
- Implement safety checks in simulation
- Validate simulation results with real-world tests
- Use simulation as a stepping stone to real robot deployment

## Summary

This chapter covered the fundamentals of Gazebo simulation for humanoid robotics, including setup, configuration, sensor integration, and control systems. Gazebo provides a powerful platform for testing humanoid robot algorithms safely and reproducibly. In the next chapter, we'll explore sensor simulation in more detail.

## Exercises

1. Create a Gazebo world with a humanoid robot model
2. Implement IMU and camera sensors on your robot
3. Configure joint controllers and test basic movement
4. Create a balance testing scenario with platform perturbations
5. Optimize your simulation for real-time performance

## Further Reading

- Gazebo Tutorials: http://gazebosim.org/tutorials
- ROS 2 Gazebo Integration: https://github.com/ros-simulation/gazebo_ros_pkgs
- Physics Simulation for Robotics: A mathematical approach

---

*Next: [Chapter 3: Sensor Simulation](../chapter3-sensor-simulation/README.md)*