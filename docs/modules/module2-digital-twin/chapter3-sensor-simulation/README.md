# Chapter 3: Sensor Simulation

## Learning Objectives

After completing this chapter, you will be able to:
- Implement and configure various sensor types in simulation environments
- Model sensor noise and limitations for realistic simulation
- Integrate sensor data with ROS 2 perception pipelines
- Design sensor fusion systems for humanoid robotics
- Validate sensor simulation against real-world performance

## Introduction to Sensor Simulation

Sensor simulation is a critical component of digital twin technology for humanoid robotics. Accurate sensor simulation allows for safe, reproducible testing of perception algorithms, sensor fusion techniques, and control systems before deployment on real robots. The goal is to create simulated sensor data that closely matches the characteristics, noise patterns, and limitations of real sensors.

### Why Sensor Simulation Matters for Humanoid Robots

Humanoid robots rely on diverse sensor modalities for navigation, balance, manipulation, and interaction:

- **Inertial sensors**: IMUs for balance and orientation
- **Vision sensors**: Cameras for environment perception and object recognition
- **Force/torque sensors**: For manipulation and contact detection
- **Range sensors**: LIDAR or depth cameras for navigation
- **Proprioceptive sensors**: Joint encoders for position feedback

Accurate simulation of these sensors is essential for developing robust humanoid robot systems.

## Types of Sensors in Humanoid Robotics

### Inertial Measurement Units (IMUs)

IMUs provide critical data for humanoid balance and orientation. In simulation, IMUs output angular velocity, linear acceleration, and sometimes magnetic field data.

#### IMU Configuration in Gazebo

```xml
<sensor name="torso_imu" type="imu">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.00001</bias_mean>
          <bias_stddev>0.000001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.00001</bias_mean>
          <bias_stddev>0.000001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
          <bias_mean>0.00001</bias_mean>
          <bias_stddev>0.000001</bias_stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.01</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.01</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
          <bias_mean>0.01</bias_mean>
          <bias_stddev>0.001</bias_stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
  <plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
    <topic_name>imu/data</topic_name>
    <body_name>torso</body_name>
    <frame_name>imu_link</frame_name>
    <initial_orientation_as_reference>false</initial_orientation_as_reference>
  </plugin>
</sensor>
```

### Camera Sensors

Cameras provide visual information for environment perception, object recognition, and navigation. In simulation, camera sensors generate realistic images with appropriate noise and distortion models.

#### Camera Configuration in Gazebo

```xml
<sensor name="head_camera" type="camera">
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
    <noise>
      <type>gaussian</type>
      <mean>0.0</mean>
      <stddev>0.007</stddev>
    </noise>
  </camera>
  <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
    <frame_name>camera_optical_frame</frame_name>
    <topic_name>camera/image_raw</topic_name>
    <camera_info_topic_name>camera/camera_info</camera_info_topic_name>
  </plugin>
</sensor>
```

### Depth Cameras

Depth cameras provide 3D information crucial for navigation and manipulation tasks:

```xml
<sensor name="depth_camera" type="depth">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <camera name="depth_camera">
    <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
    <image>
      <width>320</width>
      <height>240</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>5.0</far>
    </clip>
  </camera>
  <plugin name="depth_camera_controller" filename="libgazebo_ros_openni_kinect.so">
    <baseline>0.2</baseline>
    <alwaysOn>true</alwaysOn>
    <updateRate>30.0</updateRate>
    <cameraName>depth_camera</cameraName>
    <imageTopicName>image_raw</imageTopicName>
    <depthImageTopicName>depth/image_raw</depthImageTopicName>
    <pointCloudTopicName>points</pointCloudTopicName>
    <cameraInfoTopicName>camera_info</cameraInfoTopicName>
    <depthImageCameraInfoTopicName>depth/camera_info</depthImageCameraInfoTopicName>
    <frameName>depth_camera_optical_frame</frameName>
    <pointCloudCutoff>0.5</pointCloudCutoff>
    <pointCloudCutoffMax>3.0</pointCloudCutoffMax>
    <distortion_k1>0.0</distortion_k1>
    <distortion_k2>0.0</distortion_k2>
    <distortion_k3>0.0</distortion_k3>
    <distortion_t1>0.0</distortion_t1>
    <distortion_t2>0.0</distortion_t2>
    <CxPrime>0.0</CxPrime>
    <Cx>0.0</Cx>
    <Cy>0.0</Cy>
    <focalLength>0.0</focalLength>
    <hackBaseline>0.0</hackBaseline>
  </plugin>
</sensor>
```

### Force/Torque Sensors

Force/torque sensors are critical for manipulation and balance in humanoid robots:

```xml
<sensor name="left_foot_ft" type="force_torque">
  <always_on>true</always_on>
  <update_rate>100</update_rate>
  <force_torque>
    <frame>child</frame>
    <measure_direction>child_to_parent</measure_direction>
  </force_torque>
  <plugin name="ft_sensor_plugin" filename="libgazebo_ros_ft_sensor.so">
    <topic_name>left_foot/force_torque</topic_name>
    <joint_name>left_ankle_joint</joint_name>
    <frame_name>left_foot_frame</frame_name>
  </plugin>
</sensor>
```

### LIDAR/Ray Sensors

LIDAR sensors provide range information for navigation and mapping:

```xml
<sensor name="laser" type="ray">
  <always_on>true</always_on>
  <update_rate>10</update_rate>
  <ray>
    <scan>
      <horizontal>
        <samples>720</samples>
        <resolution>1</resolution>
        <min_angle>-2.356194</min_angle> <!-- -135 degrees -->
        <max_angle>2.356194</max_angle>  <!-- 135 degrees -->
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
    <min_range>0.1</min_range>
    <max_range>30.0</max_range>
    <gaussian_noise>0.01</gaussian_noise>
  </plugin>
</sensor>
```

## Sensor Noise Modeling

### Understanding Sensor Noise

Real sensors have various types of noise that must be modeled in simulation:

1. **Gaussian Noise**: Random variations following a normal distribution
2. **Bias**: Systematic offset in measurements
3. **Drift**: Slow changes in bias over time
4. **Quantization**: Discrete steps in digital measurements

### Noise Configuration Examples

#### IMU Noise Modeling

```xml
<imu>
  <angular_velocity>
    <x>
      <noise type="gaussian">
        <mean>0.0</mean>
        <stddev>0.001</stddev>  <!-- 1 mrad/s standard deviation -->
        <bias_mean>0.0001</bias_mean>  <!-- 0.1 mrad/s bias -->
        <bias_stddev>0.00001</bias_stddev>
        <dynamic_bias_stddev>0.00001</dynamic_bias_stddev>
        <dynamic_bias_correlation_time>100.0</dynamic_bias_correlation_time>
      </noise>
    </x>
    <!-- Similar for y and z axes -->
  </angular_velocity>
</imu>
```

#### Camera Noise Modeling

```xml
<camera>
  <!-- ... other camera parameters ... -->
  <noise>
    <type>gaussian</type>
    <mean>0.0</mean>
    <stddev>0.007</stddev>  <!-- Noise level based on real camera specs -->
  </noise>
</camera>
```

## Sensor Fusion in Simulation

### Multi-Sensor Integration

Humanoid robots typically use multiple sensors that must be fused for robust perception:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image, LaserScan, JointState
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf2_ros import TransformBroadcaster
import numpy as np
from scipy.spatial.transform import Rotation as R

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Subscriptions for different sensor types
        self.imu_sub = self.create_subscription(
            Imu, 'imu/data', self.imu_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, 'scan', self.scan_callback, 10)
        self.joint_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_callback, 10)

        # Publisher for fused state
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 'fused_pose', 10)

        # Internal state
        self.imu_data = None
        self.scan_data = None
        self.joint_data = None

        # Covariance matrices for sensor fusion
        self.imu_covariance = np.diag([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        self.odom_covariance = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    def imu_callback(self, msg):
        """Process IMU data"""
        self.imu_data = {
            'orientation': [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            'angular_velocity': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            'linear_acceleration': [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        }

    def scan_callback(self, msg):
        """Process LIDAR data"""
        self.scan_data = {
            'ranges': msg.ranges,
            'intensities': msg.intensities,
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment
        }

    def joint_callback(self, msg):
        """Process joint state data"""
        self.joint_data = {
            'names': msg.name,
            'positions': msg.position,
            'velocities': msg.velocity,
            'efforts': msg.effort
        }

    def sensor_fusion_loop(self):
        """Main sensor fusion algorithm"""
        if self.imu_data and self.joint_data:
            # Estimate pose using IMU and joint data
            pose = self.estimate_pose_from_sensors()

            # Publish fused pose
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = 'map'
            pose_msg.pose.pose.position.x = pose[0]
            pose_msg.pose.position.y = pose[1]
            pose_msg.pose.position.z = pose[2]

            # Convert orientation to quaternion
            quat = R.from_euler('xyz', [pose[3], pose[4], pose[5]]).as_quat()
            pose_msg.pose.pose.orientation.x = quat[0]
            pose_msg.pose.pose.orientation.y = quat[1]
            pose_msg.pose.pose.orientation.z = quat[2]
            pose_msg.pose.pose.orientation.w = quat[3]

            # Set covariance
            pose_msg.pose.covariance = self.calculate_covariance()

            self.pose_pub.publish(pose_msg)

    def estimate_pose_from_sensors(self):
        """Estimate pose using multiple sensor inputs"""
        # This is a simplified example
        # In practice, use an Extended Kalman Filter or similar
        if self.imu_data:
            # Extract orientation from IMU
            orientation = self.imu_data['orientation']
            euler = R.from_quat(orientation).as_euler('xyz')

            # Combine with joint positions for full pose estimation
            # (simplified for example)
            pose = [0.0, 0.0, 0.0, euler[0], euler[1], euler[2]]
            return pose
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def calculate_covariance(self):
        """Calculate covariance based on sensor fusion"""
        # Combine covariances from different sensors
        # This is a simplified approach
        combined_cov = self.imu_covariance.copy()
        return combined_cov.flatten().tolist()
```

### Extended Kalman Filter for Sensor Fusion

For more sophisticated sensor fusion, implement an Extended Kalman Filter:

```python
class EKFSensorFusion:
    def __init__(self):
        # State vector: [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
        self.state_dim = 12
        self.state = np.zeros(self.state_dim)  # State vector
        self.covariance = np.eye(self.state_dim) * 1000  # Initial uncertainty

        # Process noise
        self.Q = np.eye(self.state_dim) * 0.1

        # Measurement noise for different sensors
        self.R_imu = np.diag([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])  # orientation, angular velocity
        self.R_odom = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])  # position, orientation

    def predict(self, dt, control_input=None):
        """Prediction step of EKF"""
        # State transition model (simplified)
        F = self.calculate_jacobian_F(dt)

        # Predict state
        self.state = self.state_transition(self.state, dt, control_input)

        # Predict covariance
        self.covariance = F @ self.covariance @ F.T + self.Q

    def update_imu(self, imu_measurement):
        """Update step using IMU measurement"""
        # Measurement model
        H = self.calculate_jacobian_H_imu()
        z_pred = self.imu_measurement_model(self.state)

        # Innovation
        y = self.wrap_angles(imu_measurement - z_pred)

        # Innovation covariance
        S = H @ self.covariance @ H.T + self.R_imu

        # Kalman gain
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state = self.state + K @ y
        self.covariance = (np.eye(self.state_dim) - K @ H) @ self.covariance

    def state_transition(self, state, dt, control):
        """Nonlinear state transition function"""
        # Simplified state transition (in practice, this would be more complex)
        new_state = state.copy()

        # Update positions based on velocities
        new_state[0] += state[6] * dt  # x += vx * dt
        new_state[1] += state[7] * dt  # y += vy * dt
        new_state[2] += state[8] * dt  # z += vz * dt

        # Update orientations based on angular velocities
        new_state[3] += state[9] * dt  # roll += wx * dt
        new_state[4] += state[10] * dt  # pitch += wy * dt
        new_state[5] += state[11] * dt  # yaw += wz * dt

        return new_state

    def calculate_jacobian_F(self, dt):
        """Calculate Jacobian of state transition function"""
        F = np.eye(self.state_dim)

        # Linearized state transition
        # F[i, j] = partial derivative of state i with respect to state j
        # Simplified for this example
        F[0, 6] = dt  # dx/dvx
        F[1, 7] = dt  # dy/dvy
        F[2, 8] = dt  # dz/dvz
        F[3, 9] = dt  # droll/dwx
        F[4, 10] = dt  # dpitch/dwy
        F[5, 11] = dt  # dyaw/dwz

        return F

    def wrap_angles(self, angles):
        """Wrap angle errors to [-π, π]"""
        for i in range(len(angles)):
            while angles[i] > np.pi:
                angles[i] -= 2 * np.pi
            while angles[i] < -np.pi:
                angles[i] += 2 * np.pi
        return angles
```

## Vision Sensor Simulation

### Camera Calibration in Simulation

Simulate realistic camera parameters including distortion:

```xml
<sensor name="calibrated_camera" type="camera">
  <camera name="calibrated_camera">
    <horizontal_fov>1.047</horizontal_fov> <!-- 60 degrees -->
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>10.0</far>
    </clip>
    <distortion>
      <k1>-0.123456</k1>
      <k2>0.456789</k2>
      <k3>-0.000123</k3>
      <p1>0.000456</p1>
      <p2>-0.000789</p2>
      <center>0.5 0.5</center>
    </distortion>
  </camera>
</sensor>
```

### Stereo Vision Simulation

For depth estimation using stereo vision:

```xml
<!-- Left camera -->
<sensor name="stereo_left" type="camera">
  <pose>0.05 0.0 0.0 0 0 0</pose>
  <camera name="stereo_left">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
    </image>
  </camera>
  <plugin name="left_camera" filename="libgazebo_ros_camera.so">
    <frame_name>stereo_left_optical_frame</frame_name>
    <topic_name>stereo/left/image_raw</topic_name>
  </plugin>
</sensor>

<!-- Right camera (baseline of 10cm) -->
<sensor name="stereo_right" type="camera">
  <pose>0.05 -0.1 0.0 0 0 0</pose>
  <camera name="stereo_right">
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
    </image>
  </camera>
  <plugin name="right_camera" filename="libgazebo_ros_camera.so">
    <frame_name>stereo_right_optical_frame</frame_name>
    <topic_name>stereo/right/image_raw</topic_name>
  </plugin>
</sensor>
```

## Force and Tactile Sensing

### Contact Sensors for Tactile Feedback

```xml
<sensor name="left_foot_contact" type="contact">
  <always_on>true</always_on>
  <update_rate>1000</update_rate>
  <contact>
    <collision>left_foot_collision</collision>
  </contact>
  <plugin name="left_foot_contact_plugin" filename="libgazebo_ros_bumper.so">
    <frame_name>left_foot</frame_name>
    <topic_name>left_foot_contact</topic_name>
  </plugin>
</sensor>
```

### Multi-Contact Force Sensing

For more detailed contact information:

```xml
<!-- Define multiple contact sensors on foot for pressure distribution -->
<sensor name="left_foot_front_contact" type="contact">
  <always_on>true</always_on>
  <update_rate>1000</update_rate>
  <contact>
    <collision>left_foot_front_collision</collision>
  </contact>
</sensor>

<sensor name="left_foot_rear_contact" type="contact">
  <always_on>true</always_on>
  <update_rate>1000</update_rate>
  <contact>
    <collision>left_foot_rear_collision</collision>
  </contact>
</sensor>
```

## Sensor Validation and Verification

### Comparing Simulated vs. Real Sensors

To validate sensor simulation, compare key metrics:

```python
import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt

class SensorValidator(Node):
    def __init__(self):
        super().__init__('sensor_validator')

        # Store data for comparison
        self.sim_data = []
        self.real_data = []

        # Metrics for validation
        self.metrics = {
            'mean_error': [],
            'std_error': [],
            'correlation': [],
            'noise_characteristics': {}
        }

    def validate_imu(self, sim_imu, real_imu):
        """Compare simulated vs real IMU data"""
        # Calculate errors
        orientation_error = self.calculate_orientation_error(
            sim_imu.orientation, real_imu.orientation)
        angular_vel_error = np.linalg.norm(
            np.array(sim_imu.angular_velocity) - np.array(real_imu.angular_velocity))
        linear_acc_error = np.linalg.norm(
            np.array(sim_imu.linear_acceleration) - np.array(real_imu.linear_acceleration))

        # Store metrics
        self.metrics['mean_error'].append({
            'orientation': orientation_error,
            'angular_velocity': angular_vel_error,
            'linear_acceleration': linear_acc_error
        })

    def calculate_orientation_error(self, sim_quat, real_quat):
        """Calculate orientation error between two quaternions"""
        # Convert to rotation vectors and calculate difference
        sim_rot = R.from_quat([sim_quat.x, sim_quat.y, sim_quat.z, sim_quat.w])
        real_rot = R.from_quat([real_quat.x, real_quat.y, real_quat.z, real_quat.w])

        # Calculate relative rotation
        relative_rot = sim_rot.inv() * real_rot
        angle_error = relative_rot.magnitude()

        return angle_error

    def analyze_noise_characteristics(self, sensor_data):
        """Analyze noise characteristics of sensor data"""
        # Calculate power spectral density
        # Analyze frequency content
        # Compare noise distributions
        pass
```

## Performance Considerations

### Optimizing Sensor Simulation

Sensor simulation can be computationally expensive. Optimize performance:

```xml
<!-- Reduce update rates for less critical sensors -->
<sensor name="environment_camera" type="camera">
  <update_rate>10</update_rate>  <!-- Lower rate for environment monitoring -->
  <!-- ... -->
</sensor>

<!-- Use smaller image sizes for processing-intensive sensors -->
<camera name="processing_camera">
  <image>
    <width>320</width>  <!-- Smaller than full resolution -->
    <height>240</height>
    <format>L8</format>  <!-- Grayscale if color not needed -->
  </image>
</camera>
```

### Multi-threaded Sensor Processing

```python
import threading
from queue import Queue

class MultiThreadedSensorProcessor:
    def __init__(self):
        self.imu_queue = Queue()
        self.camera_queue = Queue()
        self.laser_queue = Queue()

        # Start processing threads
        self.imu_thread = threading.Thread(target=self.process_imu_data)
        self.camera_thread = threading.Thread(target=self.process_camera_data)
        self.laser_thread = threading.Thread(target=self.process_laser_data)

        self.imu_thread.start()
        self.camera_thread.start()
        self.laser_thread.start()

    def process_imu_data(self):
        """Process IMU data in separate thread"""
        while True:
            if not self.imu_queue.empty():
                imu_msg = self.imu_queue.get()
                # Process IMU data
                self.handle_imu_processing(imu_msg)

    def process_camera_data(self):
        """Process camera data in separate thread"""
        while True:
            if not self.camera_queue.empty():
                camera_msg = self.camera_queue.get()
                # Process camera data
                self.handle_camera_processing(camera_msg)
```

## Sensor Simulation Best Practices

### 1. Realistic Noise Modeling
- Base noise parameters on real sensor specifications
- Include bias, drift, and temperature effects
- Validate noise characteristics against real data

### 2. Computational Efficiency
- Use appropriate update rates for each sensor type
- Optimize sensor configurations for your specific use case
- Consider sensor fusion to reduce the number of required sensors

### 3. Validation and Verification
- Compare simulated sensor output with real sensors
- Validate in controlled environments before complex scenarios
- Use statistical methods to verify sensor behavior

### 4. Integration with Perception Pipelines
- Ensure sensor topics match expected formats
- Verify coordinate frame conventions
- Test sensor data quality for downstream algorithms

## Summary

This chapter covered sensor simulation for humanoid robotics, including various sensor types, noise modeling, sensor fusion techniques, and validation approaches. Accurate sensor simulation is crucial for developing robust humanoid robot perception and control systems. In the next chapter, we'll explore Unity integration for high-fidelity visualization.

## Exercises

1. Configure an IMU sensor with realistic noise parameters
2. Implement a basic sensor fusion algorithm combining IMU and joint data
3. Create a stereo vision system in simulation
4. Validate simulated sensor data against theoretical models
5. Optimize sensor simulation for computational performance

## Further Reading

- "Probabilistic Robotics" by Thrun, Burgard, and Fox
- Gazebo Sensor Documentation: http://gazebosim.org/tutorials?tut=ros_gzplugins
- Sensor Fusion Techniques for Robotics

---

*Next: [Chapter 4: Unity for High-Fidelity Robot Visualization](../chapter4-unity/README.md)*