# Chapter 5: Real-Time Control Concepts

## Learning Objectives

After completing this chapter, you will be able to:
- Understand real-time systems requirements for humanoid robotics
- Implement real-time control loops with appropriate timing
- Design feedback control systems for humanoid robots
- Apply PID control and advanced control strategies
- Ensure deterministic behavior in ROS 2 control systems

## Introduction to Real-Time Control

Real-time control is critical for humanoid robotics as these systems must respond to environmental changes and maintain balance within strict timing constraints. Unlike non-real-time systems where tasks can be delayed, humanoid robots require deterministic timing to maintain stability and safety.

### Hard vs. Soft Real-Time Systems

- **Hard Real-Time**: Missing a deadline is considered a system failure (e.g., maintaining balance)
- **Soft Real-Time**: Missing a deadline degrades performance but doesn't cause failure (e.g., path planning updates)

Humanoid robots typically require a combination of both, with critical control loops (balance, joint control) needing hard real-time guarantees.

### Timing Requirements for Humanoid Control

Different control tasks require different update frequencies:

| Control Task | Frequency | Deadline (ms) | Criticality |
|--------------|-----------|---------------|-------------|
| Joint position control | 1-2 kHz | 0.5-1 | Hard |
| Balance control | 100-500 Hz | 2-10 | Hard |
| Walking pattern generation | 10-100 Hz | 10-100 | Soft |
| Path planning | 1-10 Hz | 100-1000 | Soft |
| High-level behavior | 0.1-1 Hz | 1000-10000 | Soft |

## Real-Time Control in ROS 2

### ROS 2 Quality of Service for Control

For real-time control, configure appropriate QoS settings:

```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class RealTimeController(Node):
    def __init__(self):
        super().__init__('realtime_controller')

        # High-frequency control requires reliable, low-latency communication
        control_qos = QoSProfile(
            depth=1,  # Minimal queue to reduce latency
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            # Additional real-time settings
        )

        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, control_qos)

        self.control_pub = self.create_publisher(
            Float64MultiArray, 'control_commands', control_qos)

    def joint_state_callback(self, msg):
        # Process joint states with minimal latency
        self.process_control_logic(msg)
```

### Timing Considerations in ROS 2

```python
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
import time

class TimingAwareController(Node):
    def __init__(self):
        super().__init__('timing_aware_controller')

        # Create timer for control loop at 500Hz (2ms period)
        self.control_period = 0.002  # 2ms = 500Hz
        self.timer = self.create_timer(self.control_period, self.control_loop)

        # Track timing statistics
        self.last_execution_time = None
        self.max_execution_time = 0.0
        self.period_violations = 0

    def control_loop(self):
        start_time = time.time()

        # Your control logic here
        self.execute_control_algorithm()

        execution_time = time.time() - start_time

        # Check for timing violations
        if execution_time > self.control_period:
            self.period_violations += 1
            self.get_logger().warn(
                f'Timing violation: execution time {execution_time:.6f}s '
                f'exceeds period {self.control_period:.6f}s')

        # Track max execution time for analysis
        if execution_time > self.max_execution_time:
            self.max_execution_time = execution_time

        # Log timing statistics periodically
        if self.get_clock().now().nanoseconds % 1000000000 < 1000000:  # Every ~1 second
            self.get_logger().info(
                f'Control timing - Max: {self.max_execution_time:.6f}s, '
                f'Violations: {self.period_violations}')

    def execute_control_algorithm(self):
        # Implement your control algorithm here
        pass
```

## Control Architecture for Humanoid Robots

### Hierarchical Control Structure

Humanoid control systems typically follow a hierarchical structure:

```
High-Level Planner (1-10 Hz)
    ↓ (desired trajectories)
Trajectory Generator (10-100 Hz)
    ↓ (reference positions/velocities)
Low-Level Controller (500-1000 Hz)
    ↓ (torque/force commands)
Actuators
```

### Joint-Level Control

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

class JointController(Node):
    def __init__(self):
        super().__init__('joint_controller')

        # Joint state subscription
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.joint_state_callback, 10)

        # Control command publisher
        self.control_pub = self.create_publisher(
            Float64MultiArray, 'joint_commands', 10)

        # Store current joint states
        self.current_positions = {}
        self.current_velocities = {}
        self.current_efforts = {}

        # Control parameters
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint'
            # Add more joints as needed
        ]

        # Initialize PID controllers for each joint
        self.pid_controllers = {}
        for joint_name in self.joint_names:
            self.pid_controllers[joint_name] = PIDController(
                kp=100.0, ki=10.0, kd=5.0, max_output=100.0)

    def joint_state_callback(self, msg):
        # Update current joint states
        for i, name in enumerate(msg.name):
            if name in self.current_positions:
                self.current_positions[name] = msg.position[i]
            if name in self.current_velocities:
                self.current_velocities[name] = msg.velocity[i]
            if name in self.current_efforts:
                self.current_efforts[name] = msg.effort[i]

    def compute_control_commands(self, desired_positions, desired_velocities):
        """Compute torque/effort commands for all joints"""
        commands = Float64MultiArray()

        for joint_name in self.joint_names:
            if joint_name in self.current_positions:
                current_pos = self.current_positions[joint_name]
                current_vel = self.current_velocities.get(joint_name, 0.0)

                # Get desired values
                desired_pos = desired_positions.get(joint_name, current_pos)
                desired_vel = desired_velocities.get(joint_name, 0.0)

                # Compute control effort using PID
                effort = self.pid_controllers[joint_name].compute(
                    desired_pos, current_pos, desired_vel, current_vel)

                commands.data.append(effort)

        return commands

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, max_output=100.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.integral = 0.0
        self.previous_error = 0.0
        self.previous_time = None

    def compute(self, desired_pos, current_pos, desired_vel, current_vel):
        # Position error
        pos_error = desired_pos - current_pos

        # Velocity error
        vel_error = desired_vel - current_vel

        # Combined error (position + velocity)
        error = pos_error + 0.1 * vel_error  # Weight velocity error less

        # Get current time for derivative calculation
        current_time = time.time()
        dt = 0.002  # Assuming 500Hz control loop

        # Update integral (with anti-windup)
        self.integral += error * dt
        # Anti-windup: limit integral to prevent saturation
        self.integral = max(min(self.integral, self.max_output), -self.max_output)

        # Calculate derivative
        if self.previous_time is not None:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0.0

        # Store values for next iteration
        self.previous_error = error
        self.previous_time = current_time

        # Compute PID output
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)

        # Limit output
        output = max(min(output, self.max_output), -self.max_output)

        return output
```

## Balance Control Systems

### Center of Mass (CoM) Control

```python
import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import Float64

class BalanceController(Node):
    def __init__(self):
        super().__init__('balance_controller')

        # Subscriptions
        self.com_sub = self.create_subscription(
            Point, 'center_of_mass', self.com_callback, 10)
        self.zmp_sub = self.create_subscription(
            Point, 'zero_moment_point', self.zmp_callback, 10)

        # Publishers
        self.com_error_pub = self.create_publisher(Float64, 'com_error', 10)

        # Balance control parameters
        self.com_reference = np.array([0.0, 0.0, 0.0])  # Desired CoM position
        self.com_tolerance = 0.05  # 5cm tolerance

        # Create control timer (100Hz for balance)
        self.balance_timer = self.create_timer(0.01, self.balance_control_loop)

        # Current state
        self.current_com = np.array([0.0, 0.0, 0.0])
        self.current_zmp = np.array([0.0, 0.0, 0.0])

    def com_callback(self, msg):
        self.current_com = np.array([msg.x, msg.y, msg.z])

    def zmp_callback(self, msg):
        self.current_zmp = np.array([msg.x, msg.y, msg.z])

    def balance_control_loop(self):
        # Calculate CoM error
        com_error = self.com_reference - self.current_com

        # Check if we're outside balance boundaries
        if np.linalg.norm(com_error[:2]) > self.com_tolerance:
            self.get_logger().warn(f'CoM error too large: {com_error[:2]}')
            # Trigger recovery behavior
            self.trigger_balance_recovery()

        # Publish CoM error for monitoring
        error_msg = Float64()
        error_msg.data = float(np.linalg.norm(com_error))
        self.com_error_pub.publish(error_msg)

    def trigger_balance_recovery(self):
        """Implement balance recovery strategy"""
        self.get_logger().info('Balance recovery triggered')
        # Send commands to adjust posture
        # This would interface with higher-level controllers
```

### Walking Pattern Generation

```python
class WalkingPatternGenerator:
    def __init__(self):
        self.step_length = 0.3  # meters
        self.step_width = 0.2   # meters
        self.step_height = 0.05 # meters (clearance)
        self.step_duration = 1.0 # seconds

    def generate_walking_pattern(self, num_steps, start_pos=np.array([0, 0, 0])):
        """Generate a sequence of footstep positions"""
        footsteps = []

        current_pos = start_pos.copy()

        for i in range(num_steps):
            # Alternate between left and right foot
            foot_offset = self.step_width/2 if i % 2 == 0 else -self.step_width/2

            # Calculate next step position
            next_pos = current_pos.copy()
            next_pos[0] += self.step_length  # Move forward
            next_pos[1] = foot_offset        # Side position

            footsteps.append({
                'position': next_pos,
                'step_number': i,
                'foot': 'left' if i % 2 == 0 else 'right',
                'time': i * self.step_duration
            })

            current_pos = next_pos

        return footsteps
```

## Advanced Control Strategies

### Model Predictive Control (MPC)

For complex humanoid control, Model Predictive Control can be implemented:

```python
class MPCController:
    def __init__(self, prediction_horizon=10, control_horizon=5):
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon

        # Linearized model matrices (simplified example)
        self.A = np.eye(12)  # State transition matrix
        self.B = np.zeros((12, 6))  # Control input matrix
        # These would be derived from robot dynamics model

    def compute_control(self, current_state, reference_trajectory):
        """Compute optimal control sequence using simplified MPC"""
        # This is a simplified example
        # In practice, this would solve a constrained optimization problem

        # Predict future states
        predicted_states = []
        state = current_state.copy()

        for i in range(self.prediction_horizon):
            # Apply control (simplified)
            control_input = self.compute_feedback_control(state, reference_trajectory[i])
            state = self.A @ state + self.B @ control_input
            predicted_states.append(state)

        return control_input  # Return first control in sequence

    def compute_feedback_control(self, current_state, reference_state):
        """Simple LQR-based feedback control"""
        # Simplified LQR control law: u = -K(x - x_ref)
        error = current_state - reference_state
        K = np.eye(len(current_state)) * 0.1  # Simplified gain matrix
        control = -K @ error
        return control
```

### Impedance Control

For compliant humanoid behavior:

```python
class ImpedanceController:
    def __init__(self, stiffness=1000, damping=200, mass=100):
        self.stiffness = stiffness  # N/m
        self.damping = damping      # Ns/m
        self.mass = mass           # kg

    def compute_impedance_force(self, desired_pos, current_pos, desired_vel, current_vel):
        """Compute impedance-based force command"""
        pos_error = desired_pos - current_pos
        vel_error = desired_vel - current_vel

        # F = M*a + B*v + K*x (impedance model)
        force = (self.mass * 0) + (self.damping * vel_error) + (self.stiffness * pos_error)

        return force
```

## Real-Time Implementation Considerations

### CPU Affinity and Priority

For hard real-time requirements, consider CPU affinity:

```python
import os
import psutil

def set_realtime_priority():
    """Set process priority for real-time performance"""
    # Get the current process
    p = psutil.Process(os.getpid())

    # Set high priority (be careful with this!)
    try:
        p.nice(-10)  # High priority on Unix systems
    except psutil.AccessDenied:
        print("Could not set high priority - requires elevated privileges")

def set_cpu_affinity():
    """Pin process to specific CPU core"""
    p = psutil.Process(os.getpid())

    # Use only CPU core 1 for real-time tasks
    # This helps avoid context switching overhead
    try:
        p.cpu_affinity([1])
    except psutil.AccessDenied:
        print("Could not set CPU affinity - requires elevated privileges")
```

### Memory Management

Prevent memory allocation during real-time loops:

```python
class PreallocatedController:
    def __init__(self):
        # Pre-allocate all arrays that will be used in real-time loop
        self.state_vector = np.zeros(12, dtype=np.float64)
        self.control_output = np.zeros(6, dtype=np.float64)
        self.error_buffer = np.zeros(100, dtype=np.float64)  # For integral terms

    def real_time_loop(self):
        # This function should not allocate new memory
        # Use pre-allocated arrays only
        pass
```

## Control Safety and Monitoring

### Safety Monitoring

```python
class SafetyMonitor(Node):
    def __init__(self):
        super().__init__('safety_monitor')

        # Safety thresholds
        self.joint_position_limits = {
            'hip_joint': (-1.57, 1.57),
            'knee_joint': (0.0, 2.0),
            'ankle_joint': (-0.78, 0.78)
        }

        self.joint_velocity_limits = {joint: 5.0 for joint in self.joint_position_limits.keys()}
        self.joint_effort_limits = {joint: 100.0 for joint in self.joint_position_limits.keys()}

        # Monitor joint states
        self.joint_state_sub = self.create_subscription(
            JointState, 'joint_states', self.safety_check, 10)

        # Emergency stop publisher
        self.emergency_stop_pub = self.create_publisher(Bool, 'emergency_stop', 10)

        # Create timer for safety checks (1kHz)
        self.safety_timer = self.create_timer(0.001, self.periodic_safety_check)

        self.emergency_stop_active = False

    def safety_check(self, msg):
        """Check if any joint is in unsafe state"""
        if self.emergency_stop_active:
            return

        for i, joint_name in enumerate(msg.name):
            if joint_name in self.joint_position_limits:
                # Check position limits
                pos = msg.position[i]
                min_pos, max_pos = self.joint_position_limits[joint_name]
                if pos < min_pos or pos > max_pos:
                    self.trigger_emergency_stop(f"Joint {joint_name} position limit exceeded")
                    return

                # Check velocity limits
                vel = msg.velocity[i]
                if abs(vel) > self.joint_velocity_limits[joint_name]:
                    self.trigger_emergency_stop(f"Joint {joint_name} velocity limit exceeded")
                    return

                # Check effort limits
                effort = msg.effort[i]
                if abs(effort) > self.joint_effort_limits[joint_name]:
                    self.trigger_emergency_stop(f"Joint {joint_name} effort limit exceeded")
                    return

    def periodic_safety_check(self):
        """Additional periodic safety checks"""
        # Check for system health
        # Monitor CPU usage
        # Check communication timeouts
        pass

    def trigger_emergency_stop(self, reason):
        """Activate emergency stop"""
        self.get_logger().error(f"EMERGENCY STOP: {reason}")
        self.emergency_stop_active = True

        # Publish emergency stop command
        stop_msg = Bool()
        stop_msg.data = True
        self.emergency_stop_pub.publish(stop_msg)
```

## Performance Optimization

### Efficient Control Loop Implementation

```python
class OptimizedController(Node):
    def __init__(self):
        super().__init__('optimized_controller')

        # Use efficient data structures
        self.joint_data = {}  # Dictionary for O(1) access
        self.control_commands = []  # Pre-sized list if possible

        # Minimize function calls in critical loop
        self._control_frequency = 500  # Hz
        self._control_period = 1.0 / self._control_frequency

        # Create timer
        self.control_timer = self.create_timer(
            self._control_period, self.optimized_control_loop)

    def optimized_control_loop(self):
        """Optimized control loop with minimal overhead"""
        # Inline critical calculations
        # Avoid function calls when possible
        # Use efficient numpy operations

        # Example: simple PD control
        for joint_name, joint_info in self.joint_data.items():
            error = joint_info['desired_pos'] - joint_info['current_pos']
            vel_error = joint_info['desired_vel'] - joint_info['current_vel']

            # Inline PD calculation
            effort = (joint_info['kp'] * error) + (joint_info['kd'] * vel_error)

            # Apply limits inline
            effort = max(min(effort, joint_info['max_effort']), -joint_info['max_effort'])

            joint_info['output'] = effort
```

## Summary

This chapter covered real-time control concepts essential for humanoid robotics, including timing requirements, control architectures, balance systems, and safety considerations. Real-time control is fundamental to humanoid robot stability and safety, requiring careful attention to timing, system architecture, and safety measures.

## Exercises

1. Implement a simple PID controller for a humanoid robot joint
2. Create a balance controller that maintains center of mass within support polygon
3. Design a walking pattern generator for bipedal locomotion
4. Implement safety monitoring for joint position, velocity, and effort limits
5. Create a hierarchical control system with different update frequencies

## Further Reading

- Real-Time Systems for Robotics: https://www.springer.com/gp/book/9783319566956
- ROS 2 Real-Time Programming: https://docs.ros.org/en/rolling/How-To-Guides/Real-Time-Programming.html
- Control Systems in Robotics: Spong, Hutchinson, and Vidyasagar's Robot Modeling and Control

---

*Next: [Module 2 Preface](../preface.md)*