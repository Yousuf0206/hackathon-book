# Chapter 4: Robot Description Formats (URDF for humanoid robots)

## Learning Objectives

After completing this chapter, you will be able to:
- Create and structure URDF (Unified Robot Description Format) files for humanoid robots
- Define robot kinematics and dynamics using URDF
- Implement visual and collision models for humanoid robots
- Use Xacro for parameterized and modular robot descriptions
- Integrate URDF with ROS 2 simulation and visualization tools

## Introduction to URDF

Unified Robot Description Format (URDF) is an XML-based format used to describe robots in ROS. For humanoid robots, URDF provides a standardized way to define the robot's physical structure, including links (rigid bodies), joints (kinematic constraints), and other properties like visual appearance and collision geometry.

### Why URDF for Humanoid Robots?

- **Standardization**: Common format across the ROS ecosystem
- **Simulation Integration**: Direct compatibility with Gazebo and other simulators
- **Visualization**: Works with RViz for robot visualization
- **Kinematics**: Enables forward and inverse kinematics calculations
- **Dynamics**: Provides parameters for physics simulation

## URDF Structure for Humanoid Robots

### Basic URDF Components

A humanoid robot URDF consists of:

1. **Links**: Rigid bodies representing robot parts (torso, head, limbs)
2. **Joints**: Connections between links (revolute, prismatic, fixed)
3. **Visual Elements**: How the robot appears in visualization
4. **Collision Elements**: Geometry used for collision detection
5. **Inertial Properties**: Mass, center of mass, and inertia for dynamics

### Basic URDF Example

Here's a simple humanoid robot structure:

```xml
<?xml version="1.0"?>
<robot name="simple_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="10.0"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
    </collision>
  </link>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
      <material name="white">
        <color rgba="1 1 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.15"/>
      </geometry>
    </collision>
  </link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="base_link"/>
    <child link="head"/>
    <origin xyz="0 0 1.0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
    <dynamics damping="1.0" friction="0.1"/>
  </joint>

</robot>
```

## Humanoid Robot Kinematic Structure

### Standard Humanoid Topology

A typical humanoid robot follows this kinematic structure:

```
base_link (world)
├── torso
│   ├── head
│   ├── left_shoulder
│   │   ├── left_upper_arm
│   │   │   ├── left_lower_arm
│   │   │   │   └── left_hand
│   │   │   └── left_elbow
│   │   │       └── left_lower_arm
│   │   └── left_arm
│   │       └── left_forearm
│   │           └── left_hand
│   ├── right_shoulder
│   │   └── right_arm
│   │       └── right_forearm
│   │           └── right_hand
│   ├── pelvis
│   │   ├── left_hip
│   │   │   ├── left_thigh
│   │   │   │   ├── left_shin
│   │   │   │   │   └── left_foot
│   │   │   │   └── left_knee
│   │   │   │       └── left_shin
│   │   └── right_hip
│   │       └── right_leg
│   │           └── right_lower_leg
│   │               └── right_foot
```

### Complete Humanoid URDF Example

Here's a more comprehensive humanoid robot URDF:

```xml
<?xml version="1.0"?>
<robot name="full_humanoid" xmlns:xacro="http://www.ros.org/wiki/xacro">

  <!-- Include gazebo plugins -->
  <gazebo>
    <plugin name="gazebo_ros_control" filename="libgazebo_ros_control.so">
      <robotNamespace>/humanoid</robotNamespace>
    </plugin>
  </gazebo>

  <!-- Base link -->
  <link name="base_link">
    <inertial>
      <mass value="15.0"/>
      <origin xyz="0 0 0.5"/>
      <inertia ixx="1.5" ixy="0.0" ixz="0.0" iyy="1.5" iyz="0.0" izz="1.5"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
      <material name="light_grey">
        <color rgba="0.7 0.7 0.7 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.5"/>
      <geometry>
        <box size="0.3 0.3 1.0"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso -->
  <link name="torso">
    <inertial>
      <mass value="8.0"/>
      <origin xyz="0 0 0.3"/>
      <inertia ixx="0.8" ixy="0.0" ixz="0.0" iyy="0.8" iyz="0.0" izz="0.8"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.3"/>
      <geometry>
        <box size="0.25 0.25 0.6"/>
      </geometry>
      <material name="dark_grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.3"/>
      <geometry>
        <box size="0.25 0.25 0.6"/>
      </geometry>
    </collision>
  </link>

  <!-- Torso joint -->
  <joint name="torso_joint" type="fixed">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 1.0"/>
  </joint>

  <!-- Head -->
  <link name="head">
    <inertial>
      <mass value="3.0"/>
      <origin xyz="0 0 0.05"/>
      <inertia ixx="0.05" ixy="0.0" ixz="0.0" iyy="0.05" iyz="0.0" izz="0.05"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0.05"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
      <material name="skin">
        <color rgba="0.9 0.8 0.7 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 0.05"/>
      <geometry>
        <sphere radius="0.12"/>
      </geometry>
    </collision>
  </link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.6"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.7" upper="0.7" effort="10" velocity="2.0"/>
    <dynamics damping="0.5" friction="0.1"/>
  </joint>

  <!-- Left shoulder -->
  <link name="left_shoulder">
    <inertial>
      <mass value="1.5"/>
      <origin xyz="0 0.05 0"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    <visual>
      <origin xyz="0 0.05 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0.05 0"/>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_shoulder_joint" type="revolute">
    <parent link="torso"/>
    <child link="left_shoulder"/>
    <origin xyz="0.1 0 0.4"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" effort="50" velocity="3.0"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <!-- Left upper arm -->
  <link name="left_upper_arm">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0 0 -0.15"/>
      <inertia ixx="0.02" ixy="0.0" ixz="0.0" iyy="0.02" iyz="0.0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <origin xyz="0 0 -0.15"/>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
    </collision>
  </link>

  <joint name="left_upper_arm_joint" type="revolute">
    <parent link="left_shoulder"/>
    <child link="left_upper_arm"/>
    <origin xyz="0 0.05 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-2.0" upper="2.0" effort="50" velocity="3.0"/>
    <dynamics damping="1.0" friction="0.2"/>
  </joint>

  <!-- Additional links and joints for complete humanoid would continue... -->

</robot>
```

## Xacro for Parameterized Robot Descriptions

Xacro (XML Macros) allows you to create parameterized and modular robot descriptions, which is especially useful for humanoid robots with symmetrical components.

### Basic Xacro Example

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="xacro_humanoid">

  <!-- Define constants -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="torso_height" value="0.6" />
  <xacro:property name="torso_width" value="0.25" />
  <xacro:property name="torso_depth" value="0.25" />

  <!-- Macro for creating arm joints -->
  <xacro:macro name="arm_joint" params="side parent xyz">
    <joint name="${side}_shoulder_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_shoulder"/>
      <origin xyz="${xyz}"/>
      <axis xyz="0 0 1"/>
      <limit lower="-1.57" upper="1.57" effort="50" velocity="3.0"/>
      <dynamics damping="1.0" friction="0.2"/>
    </joint>
  </xacro:macro>

  <!-- Macro for creating leg links -->
  <xacro:macro name="leg_link" params="side position">
    <link name="${side}_leg">
      <inertial>
        <mass value="5.0"/>
        <origin xyz="0 0 -0.25"/>
        <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
      </inertial>
      <visual>
        <origin xyz="0 0 -0.25"/>
        <geometry>
          <cylinder radius="0.06" length="0.5"/>
        </geometry>
        <material name="red">
          <color rgba="1 0 0 1"/>
        </material>
      </visual>
      <collision>
        <origin xyz="0 0 -0.25"/>
        <geometry>
          <cylinder radius="0.06" length="0.5"/>
        </geometry>
      </collision>
    </link>
  </xacro:macro>

  <!-- Use macros to create symmetrical components -->
  <xacro:arm_joint side="left" parent="torso" xyz="0.1 0 0.4"/>
  <xacro:arm_joint side="right" parent="torso" xyz="-0.1 0 0.4"/>

  <xacro:leg_link side="left" position="0.05 0 -0.3"/>
  <xacro:leg_link side="right" position="-0.05 0 -0.3"/>

</robot>
```

## Advanced URDF Features for Humanoid Robots

### Transmission Elements

For simulation and control, define transmission elements:

```xml
<!-- Transmission for joint control -->
<transmission name="left_elbow_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_elbow_joint">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_elbow_motor">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Gazebo-Specific Elements

Add Gazebo-specific configurations:

```xml
<!-- Gazebo material -->
<gazebo reference="head">
  <material>Gazebo/Blue</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
  <kp>1000000.0</kp>
  <kd>100.0</kd>
</gazebo>
```

### Sensor Integration

Include sensors in the URDF:

```xml
<!-- IMU sensor -->
<gazebo reference="torso">
  <sensor name="imu_sensor" type="imu">
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
  </sensor>
</gazebo>
```

## Kinematics and Dynamics Considerations

### Center of Mass and Stability

For humanoid robots, center of mass is critical for stability:

```xml
<!-- Example with accurate CoM for balance -->
<link name="torso">
  <inertial>
    <mass value="8.0"/>
    <origin xyz="0 0 0.2"/>  <!-- Lower CoM for better stability -->
    <inertia ixx="0.8" ixy="0.0" ixz="0.0" iyy="0.8" iyz="0.0" izz="0.8"/>
  </inertial>
  <!-- ... visual and collision elements ... -->
</link>
```

### Joint Limits and Safety

Proper joint limits are essential for humanoid safety:

```xml
<!-- Hip joint with realistic limits -->
<joint name="left_hip_joint" type="revolute">
  <parent link="torso"/>
  <child link="left_thigh"/>
  <origin xyz="0.05 0 -0.1"/>
  <axis xyz="0 0 1"/>  <!-- Yaw motion -->
  <limit lower="-0.5" upper="0.5" effort="200" velocity="2.0"/>
  <dynamics damping="5.0" friction="1.0"/>
</joint>
```

## URDF Validation and Debugging

### Checking URDF Files

Use ROS tools to validate your URDF:

```bash
# Check URDF syntax
check_urdf /path/to/robot.urdf

# Parse and display robot information
urdf_to_graphiz /path/to/robot.urdf
```

### Visualization in RViz

To visualize your URDF in RViz:

1. Launch a robot state publisher:
```bash
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:=$(cat robot.urdf)
```

2. Add RobotModel display in RViz

## URDF for Different Humanoid Platforms

### NAO Humanoid Example Structure

```xml
<!-- NAO-like humanoid with 25 DOF -->
<robot name="nao_like_humanoid">
  <!-- Head with 2 DOF -->
  <link name="Head">
    <inertial>...</inertial>
    <visual>...</visual>
    <collision>...</collision>
  </link>

  <!-- Left Arm with 6 DOF -->
  <link name="LShoulderPitch">...</link>
  <link name="LShoulderRoll">...</link>
  <link name="LElbowYaw">...</link>
  <link name="LElbowRoll">...</link>
  <link name="LWristYaw">...</link>
  <link name="LHand">...</link>

  <!-- Right Arm with 6 DOF -->
  <!-- ... similar structure ... -->

  <!-- Legs with 6 DOF each -->
  <!-- ... similar structure ... -->
</robot>
```

### ATLAS Humanoid Example Structure

```xml
<!-- ATLAS-like humanoid with high DOF -->
<robot name="atlas_like_humanoid">
  <!-- More complex joint structure -->
  <!-- Additional actuators and sensors -->
  <!-- Higher fidelity collision geometry -->
</robot>
```

## Best Practices for Humanoid URDF

### 1. Modular Design
Use Xacro macros for symmetrical parts:
- Left/right arms
- Left/right legs
- Similar joint types

### 2. Accurate Physics
- Realistic mass and inertia values
- Appropriate joint limits
- Proper damping and friction

### 3. Visualization vs. Collision
- Detailed visual geometry for appearance
- Simplified collision geometry for performance
- Separate files if needed for different purposes

### 4. Consistent Naming
- Follow standard conventions (e.g., left_arm_joint)
- Use descriptive names
- Maintain consistency across similar components

## Summary

This chapter covered the fundamentals of creating URDF files for humanoid robots, including structure, kinematics, Xacro macros, and best practices. URDF is essential for humanoid robot simulation, visualization, and control. In the next chapter, we'll explore real-time control concepts for humanoid robots.

## Exercises

1. Create a complete URDF for a simple humanoid robot with at least 12 DOF
2. Use Xacro to create parameterized definitions for symmetrical components
3. Add realistic joint limits and dynamics parameters for a walking humanoid
4. Integrate IMU and force/torque sensors in the URDF
5. Validate your URDF using ROS tools and visualize it in RViz

## Further Reading

- URDF Tutorials: http://wiki.ros.org/urdf/Tutorials
- Xacro Documentation: http://wiki.ros.org/xacro
- Robotics Stack Exchange: URDF tag for specific questions

---

*Next: [Chapter 5: Real-Time Control Concepts](../chapter5-control/README.md)*