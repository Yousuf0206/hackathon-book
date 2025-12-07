# Chapter 2: ROS 2 Architecture

## Learning Objectives

After completing this chapter, you will be able to:
- Describe the core architectural components of ROS 2
- Explain the communication patterns between nodes, topics, services, and actions
- Implement basic node communication in ROS 2
- Understand Quality of Service (QoS) settings and their impact
- Design node architectures for humanoid robotics applications

## ROS 2 Architecture Overview

ROS 2 (Robot Operating System 2) is not an operating system but rather a middleware framework that provides a collection of libraries, tools, and conventions for building robotic applications. The architecture is designed to address the limitations of ROS 1, particularly in areas of security, real-time performance, and multi-robot systems.

### Key Architectural Improvements

ROS 2 introduces several key architectural improvements over ROS 1:

1. **DDS-based Communication**: Uses Data Distribution Service (DDS) as the underlying communication middleware
2. **Enhanced Security**: Built-in security features including authentication, encryption, and access control
3. **Real-time Support**: Better support for real-time systems with deterministic behavior
4. **Multi-robot Support**: Improved capabilities for multi-robot systems
5. **Quality of Service (QoS)**: Configurable reliability and performance options

## Core Components

### Nodes

A node is a fundamental unit of computation in ROS 2. Each node represents a single process that performs specific functions. Nodes can be written in different programming languages (C++, Python, etc.) and communicate with each other through the ROS 2 communication layer.

#### Node Implementation

In Python, nodes are implemented by inheriting from `rclpy.Node`:

```python
import rclpy
from rclpy.node import Node

class MinimalNode(Node):
    def __init__(self):
        super().__init__('minimal_node')
        self.get_logger().info('Minimal node initialized')

def main(args=None):
    rclpy.init(args=args)
    node = MinimalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Topics and Publishers/Subscribers

Topics provide asynchronous, many-to-many communication between nodes using a publish-subscribe pattern. Publishers send messages to topics, and subscribers receive messages from topics.

#### Publisher Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    minimal_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

#### Subscriber Example

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f'I heard: "{msg.data}"')

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = MinimalSubscriber()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Services

Services provide synchronous, request-response communication between nodes. A service client sends a request to a service server, which processes the request and returns a response.

#### Service Example

Service definition (`srv/AddTwoInts.srv`):
```
int64 a
int64 b
---
int64 sum
```

Service server:
```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info(f'Incoming request\na: {request.a}, b: {request.b}')
        return response

def main(args=None):
    rclpy.init(args=args)
    minimal_service = MinimalService()
    rclpy.spin(minimal_service)
    rclpy.shutdown()

if __init__ == '__main__':
    main()
```

Service client:
```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts

class MinimalClient(Node):
    def __init__(self):
        super().__init__('minimal_client')
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = AddTwoInts.Request()

    def send_request(self, a, b):
        self.req.a = a
        self.req.b = b
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    minimal_client = MinimalClient()
    response = minimal_client.send_request(1, 2)
    minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')
    minimal_client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Actions

Actions provide asynchronous, goal-oriented communication with feedback and status updates. They are ideal for long-running tasks that require progress monitoring.

#### Action Example

Action definition (`action/Fibonacci.action`):
```
int32 order
---
int32[] sequence
---
int32[] partial_sequence
```

Action server:
```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from example_interfaces.action import Fibonacci

class MinimalActionServer(Node):
    def __init__(self):
        super().__init__('minimal_action_server')
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

    def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')

        feedback_msg = Fibonacci.Feedback()
        feedback_msg.partial_sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                return Fibonacci.Result()

            feedback_msg.partial_sequence.append(
                feedback_msg.partial_sequence[i] + feedback_msg.partial_sequence[i-1])

            goal_handle.publish_feedback(feedback_msg)
            time.sleep(1)

        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.partial_sequence
        return result

def main(args=None):
    rclpy.init(args=args)
    minimal_action_server = MinimalActionServer()
    rclpy.spin(minimal_action_server)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Quality of Service (QoS) Profiles

QoS profiles allow you to configure communication behavior based on your application's requirements. Key QoS settings include:

### Reliability
- **Reliable**: All messages are delivered (with retries)
- **Best Effort**: Messages may be lost but with lower latency

### Durability
- **Transient Local**: Late-joining subscribers receive last published value
- **Volatile**: Only new messages are received by subscribers

### History
- **Keep Last**: Maintain only the most recent N messages
- **Keep All**: Maintain all messages (limited by resource constraints)

### Example QoS Configuration
```python
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

# For critical data requiring reliability
qos_profile = QoSProfile(
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSHistoryPolicy.TRANSIENT_LOCAL
)

publisher = node.create_publisher(String, 'critical_topic', qos_profile)
```

## Parameter System

ROS 2 provides a dynamic parameter system that allows runtime configuration of nodes:

```python
import rclpy
from rclpy.node import Node

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'default_robot')
        self.declare_parameter('max_velocity', 1.0)

        # Get parameter values
        robot_name = self.get_parameter('robot_name').value
        max_velocity = self.get_parameter('max_velocity').value

        self.get_logger().info(f'Robot: {robot_name}, Max velocity: {max_velocity}')

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Lifecycle Nodes

Lifecycle nodes provide a structured approach to node state management, particularly important for safety-critical systems:

```python
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState
from rclpy.lifecycle import TransitionCallbackReturn

class LifecycleNodeExample(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_node')

    def on_configure(self, state: LifecycleState):
        self.get_logger().info('Configuring')
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState):
        self.get_logger().info('Activating')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState):
        self.get_logger().info('Deactivating')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState):
        self.get_logger().info('Cleaning up')
        return TransitionCallbackReturn.SUCCESS
```

## Architecture for Humanoid Robotics

When designing ROS 2 systems for humanoid robots, consider these architectural patterns:

### Sensor Integration Layer
- IMU, force/torque sensors, cameras, LIDAR
- Sensor fusion nodes for state estimation
- Calibration and preprocessing nodes

### Control Layer
- Joint controllers for each actuator
- Whole-body controllers for coordination
- Trajectory generators and planners

### Perception Layer
- Object detection and recognition
- Environment mapping
- Human-robot interaction modules

### Planning Layer
- Motion planning for navigation and manipulation
- Task planning for complex behaviors
- Path optimization modules

## Best Practices

1. **Node Granularity**: Balance between too many small nodes (overhead) and too few large nodes (coupling)
2. **Topic Naming**: Use descriptive, consistent naming conventions
3. **Message Design**: Keep messages efficient and well-structured
4. **Error Handling**: Implement robust error handling and recovery mechanisms
5. **Resource Management**: Properly manage memory and computational resources

## Summary

This chapter covered the core architectural components of ROS 2, including nodes, topics, services, actions, and QoS profiles. Understanding these concepts is essential for designing effective robotic systems. In the next chapter, we'll explore practical implementation of ROS 2 packages with rclpy.

## Exercises

1. Create a publisher-subscriber pair that communicates sensor data (e.g., temperature readings)
2. Implement a service that calculates the distance between two 3D points
3. Design a node architecture for a simple humanoid robot with sensors and actuators
4. Configure different QoS profiles for various types of robot data (critical vs. non-critical)

## Further Reading

- ROS 2 Design: https://design.ros2.org/
- DDS Specification: https://www.omg.org/spec/DDS/
- ROS 2 Tutorials: https://docs.ros.org/en/humble/Tutorials.html

---

*Next: [Chapter 3: Building ROS 2 Packages with rclpy](../chapter3-packages/README.md)*