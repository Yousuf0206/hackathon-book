# Chapter 4: Navigation & Path Planning

## Learning Objectives

After completing this chapter, you will be able to:
- Implement navigation systems for humanoid robots using Isaac ROS packages
- Configure and tune navigation parameters for human environments
- Plan paths using various algorithms (A*, Dijkstra, RRT)
- Handle dynamic obstacles and replanning scenarios
- Integrate perception data with navigation systems

## Introduction to Humanoid Navigation

Navigation for humanoid robots presents unique challenges compared to wheeled robots. Humanoid robots must navigate 3D environments while maintaining balance, considering their bipedal locomotion dynamics, and operating in human-centric spaces designed for legged locomotion. The navigation system must account for the robot's physical constraints, balance requirements, and the need to operate safely around humans.

### Key Challenges in Humanoid Navigation

1. **Balance Constraints**: Robot must maintain stability during movement
2. **3D Navigation**: Consider height, stairs, and vertical obstacles
3. **Dynamic Stability**: Plan for center of mass movement
4. **Human Environments**: Navigate spaces designed for humans
5. **Social Navigation**: Consider human comfort and safety

## Isaac ROS Navigation System

### Overview of Isaac ROS Navigation

Isaac ROS provides navigation capabilities specifically designed to work with NVIDIA's GPU acceleration and AI capabilities. The navigation stack includes:

- **Costmap Management**: 2D and 3D costmap generation
- **Path Planning**: Global and local planners
- **Controller Integration**: Trajectory generation and execution
- **Perception Integration**: Real-time obstacle detection and avoidance

### Navigation Stack Architecture

```
Navigation System
├── Global Planner
│   ├── A* / Dijkstra / RRT*
│   ├── Static Map Integration
│   └── Goal Management
├── Local Planner
│   ├── Trajectory Rollout
│   ├── Obstacle Avoidance
│   └── Dynamic Window Approach
├── Controller
│   ├── Footstep Planning
│   ├── Balance Control
│   └── Trajectory Following
├── Costmap
│   ├── Static Obstacles
│   ├── Dynamic Obstacles
│   └── Inflation Layer
└── Perception Interface
    ├── LIDAR Processing
    ├── Vision Integration
    └── Sensor Fusion
```

### Basic Navigation Node Implementation

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import Header
from tf2_ros import TransformListener, Buffer
import numpy as np
from scipy.spatial import distance
import math

class IsaacHumanoidNavigation(Node):
    def __init__(self):
        super().__init__('isaac_humanoid_navigation')

        # Initialize TF listener for robot pose
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Navigation state
        self.current_pose = None
        self.goal_pose = None
        self.global_path = None
        self.local_plan = None
        self.is_navigating = False

        # Subscriptions
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.path_pub = self.create_publisher(Path, '/global_plan', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)

        # Service clients
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10
        )

        # Navigation parameters
        self.declare_parameter('planner_frequency', 5.0)
        self.declare_parameter('controller_frequency', 20.0)
        self.declare_parameter('max_vel_x', 0.5)
        self.declare_parameter('min_vel_x', 0.1)
        self.declare_parameter('max_vel_theta', 1.0)
        self.declare_parameter('min_in_place_vel_theta', 0.4)
        self.declare_parameter('xy_goal_tolerance', 0.2)
        self.declare_parameter('yaw_goal_tolerance', 0.1)

        # Timers
        self.planner_timer = self.create_timer(
            1.0 / self.get_parameter('planner_frequency').value,
            self.plan_path
        )
        self.controller_timer = self.create_timer(
            1.0 / self.get_parameter('controller_frequency').value,
            self.execute_control
        )

        # Costmap and planners
        self.costmap = Costmap2D()
        self.global_planner = GlobalPlanner()
        self.local_planner = LocalPlanner()

        self.get_logger().info('Isaac Humanoid Navigation node initialized')

    def odom_callback(self, msg):
        """Update robot pose from odometry"""
        self.current_pose = msg.pose.pose

    def scan_callback(self, msg):
        """Process laser scan for obstacle detection"""
        # Update costmap with laser scan data
        self.costmap.update_with_scan(msg)

    def map_callback(self, msg):
        """Update static map"""
        self.costmap.update_static_map(msg)

    def goal_callback(self, msg):
        """Handle new navigation goal"""
        self.goal_pose = msg.pose
        self.is_navigating = True
        self.get_logger().info(f'New goal received: ({msg.pose.position.x}, {msg.pose.position.y})')

    def plan_path(self):
        """Plan global path to goal"""
        if not self.is_navigating or self.current_pose is None or self.goal_pose is None:
            return

        # Plan path using global planner
        path = self.global_planner.plan(
            self.current_pose,
            self.goal_pose,
            self.costmap
        )

        if path:
            self.global_path = path
            self.publish_path(path)
        else:
            self.get_logger().warn('Failed to plan global path')

    def execute_control(self):
        """Execute local planning and control"""
        if not self.is_navigating or self.global_path is None:
            return

        # Get local plan
        local_plan = self.local_planner.plan(
            self.current_pose,
            self.global_path,
            self.costmap
        )

        if local_plan:
            self.local_plan = local_plan
            self.publish_local_plan(local_plan)

            # Generate velocity commands
            cmd_vel = self.generate_velocity_commands(local_plan)
            self.cmd_vel_pub.publish(cmd_vel)

            # Check if goal reached
            if self.is_goal_reached():
                self.navigation_done()

    def generate_velocity_commands(self, local_plan):
        """Generate velocity commands from local plan"""
        cmd_vel = Twist()

        if len(local_plan.poses) == 0:
            return cmd_vel

        # Simple proportional controller
        target_pose = local_plan.poses[0].pose
        current_pos = self.current_pose.position
        target_pos = target_pose.position

        # Calculate distance and angle to next waypoint
        dx = target_pos.x - current_pos.x
        dy = target_pos.y - current_pos.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Calculate angle to target
        target_angle = math.atan2(dy, dx)
        current_yaw = self.get_yaw_from_quaternion(self.current_pose.orientation)
        angle_diff = target_angle - current_yaw
        angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))  # Normalize

        # Velocity commands
        max_vel = self.get_parameter('max_vel_x').value
        min_vel = self.get_parameter('min_vel_x').value
        max_theta = self.get_parameter('max_vel_theta').value

        cmd_vel.linear.x = min(max_vel, max(min_vel, distance * 1.0))  # Proportional
        cmd_vel.angular.z = max(min(max_theta, angle_diff * 2.0), -max_theta)

        return cmd_vel

    def is_goal_reached(self):
        """Check if goal has been reached"""
        if self.current_pose is None or self.goal_pose is None:
            return False

        current_pos = self.current_pose.position
        goal_pos = self.goal_pose.position

        distance = math.sqrt(
            (current_pos.x - goal_pos.x)**2 +
            (current_pos.y - goal_pos.y)**2
        )

        goal_tolerance = self.get_parameter('xy_goal_tolerance').value
        return distance <= goal_tolerance

    def navigation_done(self):
        """Handle completion of navigation"""
        self.is_navigating = False
        self.global_path = None
        self.local_plan = None

        # Stop robot
        stop_cmd = Twist()
        self.cmd_vel_pub.publish(stop_cmd)

        self.get_logger().info('Navigation goal reached')

    def publish_path(self, path):
        """Publish global path"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        path_msg.poses = path.poses
        self.path_pub.publish(path_msg)

    def publish_local_plan(self, local_plan):
        """Publish local plan"""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        path_msg.poses = local_plan.poses
        self.local_plan_pub.publish(path_msg)

    def get_yaw_from_quaternion(self, quat):
        """Extract yaw angle from quaternion"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

def main(args=None):
    rclpy.init(args=args)
    navigator = IsaacHumanoidNavigation()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Costmap Management for Humanoid Robots

### Costmap Implementation

```python
import numpy as np
from sensor_msgs.msg import LaserScan, PointCloud2, OccupancyGrid
from geometry_msgs.msg import Point
import math

class Costmap2D:
    def __init__(self, resolution=0.05, width=40, height=40):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin_x = -width * resolution / 2
        self.origin_y = -height * resolution / 2

        # Initialize costmap
        self.costmap = np.zeros((height, width), dtype=np.uint8)
        self.static_map = np.zeros((height, width), dtype=np.uint8)

        # Cost values
        self.FREE_SPACE = 0
        self.INSCRIBED_INFLATED_OBSTACLE = 253
        self.LETHAL_OBSTACLE = 254
        self.UNKNOWN = 255

        # Inflation parameters
        self.inflation_radius = 0.5  # meters
        self.cost_scaling_factor = 10.0

    def world_to_map(self, x, y):
        """Convert world coordinates to map indices"""
        map_x = int((x - self.origin_x) / self.resolution)
        map_y = int((y - self.origin_y) / self.resolution)

        if 0 <= map_x < self.width and 0 <= map_y < self.height:
            return map_x, map_y
        else:
            return None, None

    def map_to_world(self, map_x, map_y):
        """Convert map indices to world coordinates"""
        x = map_x * self.resolution + self.origin_x
        y = map_y * self.resolution + self.origin_y
        return x, y

    def update_static_map(self, occupancy_grid):
        """Update costmap with static map"""
        # Copy static map
        self.static_map = np.array(occupancy_grid.data).reshape(
            occupancy_grid.info.height,
            occupancy_grid.info.width
        )

    def update_with_scan(self, scan):
        """Update costmap with laser scan data"""
        # Clear previous dynamic obstacles
        dynamic_mask = self.costmap > self.FREE_SPACE
        self.costmap[dynamic_mask & (self.costmap < self.LETHAL_OBSTACLE)] = self.FREE_SPACE

        # Process laser scan
        robot_x, robot_y = 0, 0  # Robot at center of costmap

        for i, range_reading in enumerate(scan.ranges):
            if not (scan.range_min <= range_reading <= scan.range_max):
                continue

            # Calculate angle
            angle = scan.angle_min + i * scan.angle_increment

            # Calculate endpoint of laser beam
            endpoint_x = robot_x + range_reading * math.cos(angle)
            endpoint_y = robot_y + range_reading * math.sin(angle)

            # Mark endpoint as obstacle
            obs_x, obs_y = self.world_to_map(endpoint_x, endpoint_y)
            if obs_x is not None and obs_y is not None:
                self.set_cost(obs_x, obs_y, self.LETHAL_OBSTACLE)

                # Inflation around obstacle
                self.inflate_obstacle(obs_x, obs_y)

    def set_cost(self, x, y, cost):
        """Set cost at specific cell"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.costmap[y, x] = cost

    def get_cost(self, x, y):
        """Get cost at specific cell"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.costmap[y, x]
        else:
            return self.LETHAL_OBSTACLE  # Outside map is lethal

    def inflate_obstacle(self, obs_x, obs_y):
        """Inflate obstacle with radius"""
        inflation_cells = int(self.inflation_radius / self.resolution)

        for dx in range(-inflation_cells, inflation_cells + 1):
            for dy in range(-inflation_cells, inflation_cells + 1):
                new_x = obs_x + dx
                new_y = obs_y + dy

                if (0 <= new_x < self.width and 0 <= new_y < self.height):
                    # Calculate distance to obstacle
                    dist = math.sqrt(dx*dx + dy*dy) * self.resolution

                    if dist <= self.inflation_radius:
                        # Calculate cost based on distance
                        if dist < self.resolution:
                            cost = self.LETHAL_OBSTACLE
                        else:
                            # Inverse relationship with distance
                            normalized_dist = dist / self.inflation_radius
                            cost = int(self.INSCRIBED_INFLATED_OBSTACLE * (1 - normalized_dist))

                        # Only increase cost, don't decrease
                        current_cost = self.get_cost(new_x, new_y)
                        if cost > current_cost:
                            self.set_cost(new_x, new_y, cost)

    def is_free(self, x, y):
        """Check if cell is free for humanoid"""
        # For humanoid, we might need larger clear space
        # Check a 3x3 area around the point
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_x, check_y = x + dx, y + dy
                cost = self.get_cost(check_x, check_y)
                if cost >= self.INSCRIBED_INFLATED_OBSTACLE:
                    return False
        return True
```

## Path Planning Algorithms

### Global Path Planner

```python
import heapq
import numpy as np
from geometry_msgs.msg import Pose, Point
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class GlobalPlanner:
    def __init__(self):
        self.planner_type = 'astar'  # Options: 'astar', 'dijkstra', 'rrt'
        self.motion_primitives = self.generate_motion_primitives()

    def generate_motion_primitives(self):
        """Generate motion primitives for humanoid navigation"""
        # Define possible movement directions
        # For humanoid, consider step constraints
        primitives = []

        # Forward steps
        for step_size in [0.1, 0.2, 0.3, 0.4, 0.5]:
            primitives.append((step_size, 0, 0))  # Forward
            primitives.append((step_size, 0, 0.2))  # Forward + slight turn
            primitives.append((step_size, 0, -0.2))  # Forward - slight turn

        # Lateral steps (with turn)
        for step_size in [0.1, 0.2, 0.3]:
            primitives.append((0, step_size, 0))   # Left
            primitives.append((0, -step_size, 0))  # Right

        # Backward steps
        for step_size in [0.1, 0.2]:
            primitives.append((-step_size, 0, 0))  # Backward

        return primitives

    def plan(self, start_pose, goal_pose, costmap):
        """Plan path using selected algorithm"""
        if self.planner_type == 'astar':
            return self.astar_plan(start_pose, goal_pose, costmap)
        elif self.planner_type == 'dijkstra':
            return self.dijkstra_plan(start_pose, goal_pose, costmap)
        else:
            raise ValueError(f"Unknown planner type: {self.planner_type}")

    def astar_plan(self, start_pose, goal_pose, costmap):
        """A* path planning algorithm"""
        # Convert poses to map coordinates
        start_x, start_y = costmap.world_to_map(start_pose.position.x, start_pose.position.y)
        goal_x, goal_y = costmap.world_to_map(goal_pose.position.x, goal_pose.position.y)

        if start_x is None or start_y is None or goal_x is None or goal_y is None:
            return None

        # Check if start or goal is in obstacle
        if costmap.get_cost(start_x, start_y) >= costmap.INSCRIBED_INFLATED_OBSTACLE:
            print(f"Start position is in obstacle: {costmap.get_cost(start_x, start_y)}")
            return None

        if costmap.get_cost(goal_x, goal_y) >= costmap.INSCRIBED_INFLATED_OBSTACLE:
            print(f"Goal position is in obstacle: {costmap.get_cost(goal_x, goal_y)}")
            return None

        # A* algorithm
        open_set = [(0, (start_x, start_y))]  # (f_score, (x, y))
        came_from = {}
        g_score = {(start_x, start_y): 0}
        f_score = {(start_x, start_y): self.heuristic(start_x, start_y, goal_x, goal_y)}

        visited = set()

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == (goal_x, goal_y):
                # Reconstruct path
                path = self.reconstruct_path(came_from, current, costmap)
                return path

            if current in visited:
                continue

            visited.add(current)
            x, y = current

            # Check 8-connected neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    neighbor_x, neighbor_y = x + dx, y + dy

                    # Check bounds
                    if neighbor_x < 0 or neighbor_x >= costmap.width or \
                       neighbor_y < 0 or neighbor_y >= costmap.height:
                        continue

                    # Check if free space
                    if not costmap.is_free(neighbor_x, neighbor_y):
                        continue

                    # Calculate tentative g_score
                    movement_cost = math.sqrt(dx*dx + dy*dy)
                    cost = costmap.get_cost(neighbor_x, neighbor_y) / 255.0
                    tentative_g_score = g_score[(x, y)] + movement_cost + cost

                    neighbor = (neighbor_x, neighbor_y)

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(
                            neighbor_x, neighbor_y, goal_x, goal_y
                        )
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found

    def dijkstra_plan(self, start_pose, goal_pose, costmap):
        """Dijkstra's path planning algorithm"""
        # Similar to A* but without heuristic (heuristic = 0)
        start_x, start_y = costmap.world_to_map(start_pose.position.x, start_pose.position.y)
        goal_x, goal_y = costmap.world_to_map(goal_pose.position.x, goal_pose.position.y)

        if start_x is None or start_y is None or goal_x is None or goal_y is None:
            return None

        # Dijkstra algorithm (A* with heuristic = 0)
        open_set = [(0, (start_x, start_y))]  # (cost, (x, y))
        came_from = {}
        cost_so_far = {(start_x, start_y): 0}

        while open_set:
            current_cost, current = heapq.heappop(open_set)

            if current == (goal_x, goal_y):
                # Reconstruct path
                path = self.reconstruct_path(came_from, current, costmap)
                return path

            x, y = current

            # Check 8-connected neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue

                    neighbor_x, neighbor_y = x + dx, y + dy

                    # Check bounds
                    if neighbor_x < 0 or neighbor_x >= costmap.width or \
                       neighbor_y < 0 or neighbor_y >= costmap.height:
                        continue

                    # Check if free space
                    if not costmap.is_free(neighbor_x, neighbor_y):
                        continue

                    # Calculate movement cost
                    movement_cost = math.sqrt(dx*dx + dy*dy)
                    cell_cost = costmap.get_cost(neighbor_x, neighbor_y) / 255.0
                    new_cost = cost_so_far[(x, y)] + movement_cost + cell_cost

                    neighbor = (neighbor_x, neighbor_y)

                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        priority = new_cost
                        heapq.heappush(open_set, (priority, neighbor))
                        came_from[neighbor] = current

        return None  # No path found

    def heuristic(self, x1, y1, x2, y2):
        """Heuristic function for A* (Euclidean distance)"""
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def reconstruct_path(self, came_from, current, costmap):
        """Reconstruct path from came_from dictionary"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()

        # Convert to ROS Path message
        ros_path = Path()
        ros_path.header.stamp = self.get_current_time()
        ros_path.header.frame_id = 'map'

        for x, y in path:
            world_x, world_y = costmap.map_to_world(x, y)
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = world_x
            pose_stamped.pose.position.y = world_y
            pose_stamped.pose.position.z = 0.0  # Assuming 2D navigation
            ros_path.poses.append(pose_stamped)

        return ros_path

    def get_current_time(self):
        """Get current ROS time"""
        # This would be implemented in the actual node
        from builtin_interfaces.msg import Time
        return Time(sec=0, nanosec=0)
```

### Local Planner for Humanoid Navigation

```python
import numpy as np
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Path
import math

class LocalPlanner:
    def __init__(self):
        self.controller_frequency = 20.0  # Hz
        self.sim_time = 1.5  # seconds to simulate
        self.sim_granularity = 0.025  # meters between simulation points
        self.angular_sim_granularity = 0.1  # radian between simulation points

        # Robot parameters for humanoid
        self.max_vel_x = 0.5  # m/s
        self.min_vel_x = 0.05
        self.max_vel_theta = 1.0  # rad/s
        self.min_in_place_vel_theta = 0.4
        self.acc_lim_x = 2.5  # m/s^2
        self.acc_lim_theta = 3.2  # rad/s^2

        # Trajectory scoring parameters
        self.path_distance_bias = 0.6
        self.goal_distance_bias = 0.8
        self.occdist_scale = 0.01

    def plan(self, current_pose, global_plan, costmap):
        """Plan local trajectory considering obstacles"""
        if not global_plan.poses:
            return None

        # Get local goal from global plan
        local_goal = self.get_local_goal(current_pose, global_plan)

        # Generate possible trajectories
        trajectories = self.generate_trajectories(current_pose)

        # Score trajectories
        best_trajectory = self.score_trajectories(
            trajectories, current_pose, local_goal, costmap
        )

        if best_trajectory:
            # Convert trajectory to Path message
            local_path = self.trajectory_to_path(best_trajectory, current_pose)
            return local_path

        return None

    def get_local_goal(self, current_pose, global_plan):
        """Get goal point along global plan that's closest to robot"""
        robot_x = current_pose.position.x
        robot_y = current_pose.position.y

        # Find closest point on global plan
        closest_idx = 0
        min_dist = float('inf')

        for i, pose_stamped in enumerate(global_plan.poses):
            pose = pose_stamped.pose
            dist = math.sqrt(
                (pose.position.x - robot_x)**2 +
                (pose.position.y - robot_y)**2
            )
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # Return a point ahead on the path
        look_ahead = min(closest_idx + 10, len(global_plan.poses) - 1)
        return global_plan.poses[look_ahead].pose

    def generate_trajectories(self, current_pose):
        """Generate possible robot trajectories"""
        trajectories = []

        # Current state
        x = current_pose.position.x
        y = current_pose.position.y
        theta = self.get_yaw_from_quaternion(current_pose.orientation)
        vx = 0.0  # Assume starting from rest
        vy = 0.0
        vtheta = 0.0

        # Try different velocity combinations
        for vx_samp in np.linspace(self.min_vel_x, self.max_vel_x, 5):
            for vtheta_samp in np.linspace(-self.max_vel_theta, self.max_vel_theta, 5):
                # Simulate trajectory
                traj = self.simulate_trajectory(
                    x, y, theta, vx, vy, vtheta,
                    vx_samp, 0.0, vtheta_samp  # vtheta_samp for rotation
                )
                if traj:
                    trajectories.append(traj)

        return trajectories

    def simulate_trajectory(self, x, y, theta, vx, vy, vtheta,
                           goal_vx, goal_vy, goal_vtheta):
        """Simulate a trajectory with given velocities"""
        trajectory = []
        dt = 1.0 / self.controller_frequency

        # Simulate for sim_time seconds
        for t in np.arange(0, self.sim_time, dt):
            # Update velocities based on acceleration limits
            new_vx = self.limit_velocity(
                vx, goal_vx, self.acc_lim_x * dt
            )
            new_vtheta = self.limit_velocity(
                vtheta, goal_vtheta, self.acc_lim_theta * dt
            )

            # Update position
            x += new_vx * math.cos(theta) * dt
            y += new_vx * math.sin(theta) * dt
            theta += new_vtheta * dt

            # Store state
            trajectory.append((x, y, theta, new_vx, new_vtheta))

            # Update velocities for next iteration
            vx, vtheta = new_vx, new_vtheta

        return trajectory

    def limit_velocity(self, current_vel, desired_vel, max_delta):
        """Limit velocity change"""
        if desired_vel > current_vel:
            return min(desired_vel, current_vel + max_delta)
        else:
            return max(desired_vel, current_vel - max_delta)

    def score_trajectories(self, trajectories, current_pose, local_goal, costmap):
        """Score trajectories based on various criteria"""
        if not trajectories:
            return None

        best_score = float('-inf')
        best_trajectory = None

        for trajectory in trajectories:
            if not trajectory:
                continue

            # Check if trajectory collides with obstacles
            if self.trajectory_collides(trajectory, costmap):
                continue

            # Calculate scores
            path_score = self.calculate_path_score(trajectory, local_goal)
            goal_score = self.calculate_goal_score(trajectory, local_goal)
            occ_score = self.calculate_obstacle_score(trajectory, costmap)

            # Weighted combination of scores
            total_score = (
                self.path_distance_bias * path_score +
                self.goal_distance_bias * goal_score +
                self.occdist_scale * occ_score
            )

            if total_score > best_score:
                best_score = total_score
                best_trajectory = trajectory

        return best_trajectory

    def trajectory_collides(self, trajectory, costmap):
        """Check if trajectory collides with obstacles"""
        for x, y, theta, vx, vtheta in trajectory:
            map_x, map_y = costmap.world_to_map(x, y)
            if map_x is not None and map_y is not None:
                if costmap.get_cost(map_x, map_y) >= costmap.LETHAL_OBSTACLE:
                    return True
        return False

    def calculate_path_score(self, trajectory, local_goal):
        """Calculate score based on following the path"""
        if not trajectory:
            return float('-inf')

        final_x, final_y, final_theta, final_vx, final_vtheta = trajectory[-1]
        goal_x = local_goal.position.x
        goal_y = local_goal.position.y

        # Distance to goal (lower is better)
        dist_to_goal = math.sqrt((final_x - goal_x)**2 + (final_y - goal_y)**2)

        # Prefer trajectories that move toward the goal
        start_x, start_y = trajectory[0][0], trajectory[0][1]
        initial_dist = math.sqrt((start_x - goal_x)**2 + (start_y - goal_y)**2)
        final_dist = dist_to_goal

        if final_dist < initial_dist:
            return 1.0 / (1.0 + dist_to_goal)  # Higher score for closer to goal
        else:
            return -1.0  # Negative score if moving away

    def calculate_goal_score(self, trajectory, local_goal):
        """Calculate score based on reaching the goal"""
        if not trajectory:
            return float('-inf')

        final_x, final_y, final_theta, final_vx, final_vtheta = trajectory[-1]
        goal_x = local_goal.position.x
        goal_y = local_goal.position.y

        dist_to_goal = math.sqrt((final_x - goal_x)**2 + (final_y - goal_y)**2)

        # Higher score for closer to goal
        return 1.0 / (1.0 + dist_to_goal)

    def calculate_obstacle_score(self, trajectory, costmap):
        """Calculate score based on obstacle proximity"""
        if not trajectory:
            return float('-inf')

        total_cost = 0
        for x, y, theta, vx, vtheta in trajectory:
            map_x, map_y = costmap.world_to_map(x, y)
            if map_x is not None and map_y is not None:
                cost = costmap.get_cost(map_x, map_y)
                total_cost += cost

        # Lower total cost is better
        return -total_cost

    def trajectory_to_path(self, trajectory, start_pose):
        """Convert trajectory to Path message"""
        path = Path()
        path.header.stamp = self.get_current_time()
        path.header.frame_id = 'map'

        for x, y, theta, vx, vtheta in trajectory:
            pose_stamped = PoseStamped()
            pose_stamped.pose.position.x = x
            pose_stamped.pose.position.y = y
            pose_stamped.pose.position.z = 0.0

            # Convert theta to quaternion
            quat = self.yaw_to_quaternion(theta)
            pose_stamped.pose.orientation.x = quat[0]
            pose_stamped.pose.orientation.y = quat[1]
            pose_stamped.pose.orientation.z = quat[2]
            pose_stamped.pose.orientation.w = quat[3]

            path.poses.append(pose_stamped)

        return path

    def get_yaw_from_quaternion(self, quat):
        """Extract yaw from quaternion"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        """Convert yaw angle to quaternion"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return [0, 0, sy, cy]  # x, y, z, w

    def get_current_time(self):
        """Get current time (placeholder)"""
        from builtin_interfaces.msg import Time
        return Time(sec=0, nanosec=0)
```

## Dynamic Obstacle Avoidance

### Dynamic Window Approach for Humanoid Robots

```python
import numpy as np
from geometry_msgs.msg import Twist
import math

class DynamicWindowApproach:
    def __init__(self):
        # Robot parameters for humanoid
        self.max_vel_x = 0.5  # Maximum forward velocity
        self.min_vel_x = 0.05  # Minimum forward velocity
        self.max_vel_theta = 1.0  # Maximum angular velocity
        self.min_vel_theta = 0.1  # Minimum angular velocity (for in-place rotation)

        # Acceleration limits
        self.acc_lim_x = 2.5  # m/s^2
        self.acc_lim_theta = 3.2  # rad/s^2

        # Prediction time
        self.sim_time = 2.0  # seconds to simulate
        self.dt = 0.1  # time step for simulation

        # Goal and obstacle weights
        self.goal_cost_gain = 0.8
        self.obstacle_cost_gain = 0.1
        self.velocity_cost_gain = 0.1

    def calculate_velocity_commands(self, current_pose, current_velocity, goal_pose,
                                   obstacle_list, costmap):
        """Calculate optimal velocity commands using DWA"""
        # Define velocity windows
        vs = self.calculate_velocity_space(current_velocity)

        # Evaluate all velocities in the window
        best_score = float('-inf')
        best_vel = Twist()

        for vel_x, vel_theta in vs:
            # Simulate trajectory for this velocity
            trajectory = self.predict_trajectory(
                current_pose, vel_x, vel_theta, current_velocity
            )

            # Calculate scores
            to_goal_cost = self.to_goal_cost(trajectory, goal_pose)
            speed_cost = self.speed_cost(vel_x, vel_theta)
            obstacle_cost = self.obstacle_cost(trajectory, obstacle_list, costmap)

            # Weighted sum of costs
            score = (
                self.goal_cost_gain * to_goal_cost -
                self.obstacle_cost_gain * obstacle_cost -
                self.velocity_cost_gain * speed_cost
            )

            if score > best_score:
                best_score = score
                best_vel.linear.x = vel_x
                best_vel.angular.z = vel_theta

        return best_vel

    def calculate_velocity_space(self, current_velocity):
        """Calculate the dynamic velocity space"""
        # Current velocities
        current_vx, current_vtheta = current_velocity.linear.x, current_velocity.angular.z

        # Calculate velocity limits based on acceleration
        max_vx = min(
            self.max_vel_x,
            current_vx + self.acc_lim_x * self.dt
        )
        min_vx = max(
            self.min_vel_x,
            current_vx - self.acc_lim_x * self.dt
        )

        max_vtheta = min(
            self.max_vel_theta,
            current_vtheta + self.acc_lim_theta * self.dt
        )
        min_vtheta = max(
            -self.max_vel_theta,
            current_vtheta - self.acc_lim_theta * self.dt
        )

        # Generate velocity samples
        velocity_space = []
        for vx in np.linspace(min_vx, max_vx, 10):
            for vtheta in np.linspace(min_vtheta, max_vtheta, 10):
                velocity_space.append((vx, vtheta))

        return velocity_space

    def predict_trajectory(self, start_pose, vel_x, vel_theta, current_vel):
        """Predict trajectory for given velocities"""
        trajectory = []

        # Start from current pose
        x = start_pose.position.x
        y = start_pose.position.y
        theta = self.get_yaw_from_quaternion(start_pose.orientation)

        # Simulate for sim_time
        steps = int(self.sim_time / self.dt)

        for i in range(steps):
            # Update position based on velocities
            x += vel_x * math.cos(theta) * self.dt
            y += vel_x * math.sin(theta) * self.dt
            theta += vel_theta * self.dt

            # Store pose
            pose = Pose()
            pose.position.x = x
            pose.position.y = y
            pose.position.z = 0.0
            quat = self.yaw_to_quaternion(theta)
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]

            trajectory.append(pose)

        return trajectory

    def to_goal_cost(self, trajectory, goal_pose):
        """Calculate cost to goal"""
        if not trajectory:
            return float('inf')

        final_pose = trajectory[-1]
        dx = goal_pose.position.x - final_pose.position.x
        dy = goal_pose.position.y - final_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Lower distance = higher cost (since we're maximizing)
        return 1.0 / (distance + 0.001)  # Add small value to avoid division by zero

    def speed_cost(self, vel_x, vel_theta):
        """Calculate cost based on velocity"""
        # Prefer higher velocities (for efficiency)
        return math.sqrt(vel_x*vel_x + vel_theta*vel_theta)

    def obstacle_cost(self, trajectory, obstacle_list, costmap):
        """Calculate cost based on obstacle proximity"""
        if not trajectory:
            return 0

        min_dist_to_obstacle = float('inf')

        for pose in trajectory:
            # Check distance to all obstacles
            for obs in obstacle_list:
                dx = pose.position.x - obs.position.x
                dy = pose.position.y - obs.position.y
                dist = math.sqrt(dx*dx + dy*dy)

                if dist < min_dist_to_obstacle:
                    min_dist_to_obstacle = dist

        # Higher cost for closer obstacles
        if min_dist_to_obstacle < 0.001:
            return float('inf')  # Collision
        else:
            return 1.0 / min_dist_to_obstacle

    def get_yaw_from_quaternion(self, quat):
        """Extract yaw from quaternion"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def yaw_to_quaternion(self, yaw):
        """Convert yaw to quaternion"""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        return [0, 0, sy, cy]
```

## Social Navigation for Humanoid Robots

### Human-Aware Navigation

```python
import numpy as np
from geometry_msgs.msg import Pose, Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import math

class SocialNavigation:
    def __init__(self):
        # Social force parameters
        self.social_force_strength = 5.0
        self.influence_distance = 2.0  # meters
        self.body_force_strength = 1.0
        self.wall_force_strength = 2.0

        # Personal space parameters
        self.intimate_space = 0.45  # 0.45m
        self.personal_space = 1.2   # 1.2m
        self.social_space = 3.6     # 3.6m

        # Walking behavior parameters
        self.comfortable_walking_speed = 1.4  # m/s (human average)
        self.urgent_walking_speed = 2.0       # m/s

    def calculate_social_velocity(self, current_pose, current_vel, humans,
                                  goal_pose, costmap):
        """Calculate velocity considering human presence"""
        # Calculate desired velocity toward goal
        desired_vel = self.calculate_desired_velocity(current_pose, goal_pose)

        # Calculate social forces from humans
        social_force = self.calculate_social_forces(current_pose, humans)

        # Calculate wall forces
        wall_force = self.calculate_wall_forces(current_pose, costmap)

        # Combine forces
        total_force = self.add_vectors(desired_vel, social_force, weight=0.8)
        total_force = self.add_vectors(total_force, wall_force, weight=0.2)

        # Limit to maximum velocity
        max_speed = self.comfortable_walking_speed
        force_magnitude = math.sqrt(total_force.linear.x**2 + total_force.linear.y**2)

        if force_magnitude > max_speed:
            scale = max_speed / force_magnitude
            total_force.linear.x *= scale
            total_force.linear.y *= scale

        return total_force

    def calculate_desired_velocity(self, current_pose, goal_pose):
        """Calculate desired velocity toward goal"""
        dx = goal_pose.position.x - current_pose.position.x
        dy = goal_pose.position.y - current_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        if distance < 0.1:  # Close to goal
            return Twist()

        # Normalize direction
        desired_x = dx / distance
        desired_y = dy / distance

        # Scale by comfortable walking speed
        vel = Twist()
        vel.linear.x = desired_x * self.comfortable_walking_speed
        vel.linear.y = desired_y * self.comfortable_walking_speed

        # Calculate angular velocity for orientation
        desired_theta = math.atan2(dy, dx)
        current_theta = self.get_yaw_from_quaternion(current_pose.orientation)
        theta_diff = desired_theta - current_theta
        theta_diff = math.atan2(math.sin(theta_diff), math.cos(theta_diff))  # Normalize

        vel.angular.z = theta_diff * 2.0  # Proportional controller

        return vel

    def calculate_social_forces(self, current_pose, humans):
        """Calculate repulsive forces from humans"""
        total_force_x = 0.0
        total_force_y = 0.0

        for human in humans:
            # Calculate distance and direction to human
            dx = human.position.x - current_pose.position.x
            dy = human.position.y - current_pose.position.y
            distance = math.sqrt(dx*dx + dy*dy)

            if distance < 0.01:  # Very close, avoid division by zero
                continue

            # Normalize direction (away from human)
            force_x = -dx / distance
            force_y = -dy / distance

            # Calculate force magnitude based on distance
            # Stronger force when closer
            if distance < self.influence_distance:
                force_magnitude = self.social_force_strength / (distance + 0.1)
                total_force_x += force_x * force_magnitude
                total_force_y += force_y * force_magnitude

        # Create force vector as Twist
        force = Twist()
        force.linear.x = total_force_x
        force.linear.y = total_force_y

        return force

    def calculate_wall_forces(self, current_pose, costmap):
        """Calculate repulsive forces from walls/obstacles"""
        # Sample points around the robot
        force_x = 0.0
        force_y = 0.0

        robot_x = current_pose.position.x
        robot_y = current_pose.position.y

        # Check in 8 directions around the robot
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            angle_rad = math.radians(angle)

            # Check for obstacles in this direction
            for dist in np.linspace(0.1, 1.0, 10):  # Check up to 1m
                check_x = robot_x + dist * math.cos(angle_rad)
                check_y = robot_y + dist * math.sin(angle_rad)

                map_x, map_y = costmap.world_to_map(check_x, check_y)

                if map_x is not None and map_y is not None:
                    if costmap.get_cost(map_x, map_y) >= costmap.INSCRIBED_INFLATED_OBSTACLE:
                        # Found obstacle, add repulsive force
                        force_x += -math.cos(angle_rad) * self.wall_force_strength / dist
                        force_y += -math.sin(angle_rad) * self.wall_force_strength / dist
                        break  # Stop checking this direction once obstacle found

        force = Twist()
        force.linear.x = force_x
        force.linear.y = force_y

        return force

    def add_vectors(self, twist1, twist2, weight=1.0):
        """Add two twist vectors with optional weighting"""
        result = Twist()
        result.linear.x = twist1.linear.x + weight * twist2.linear.x
        result.linear.y = twist1.linear.y + weight * twist2.linear.y
        result.angular.z = twist1.angular.z + weight * twist2.angular.z

        return result

    def get_yaw_from_quaternion(self, quat):
        """Extract yaw from quaternion"""
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def should_yield_to_human(self, current_pose, human_pose, human_vel):
        """Determine if robot should yield to human"""
        # Calculate distance to human
        dx = human_pose.position.x - current_pose.position.x
        dy = human_pose.position.y - current_pose.position.y
        distance = math.sqrt(dx*dx + dy*dy)

        # Check if in social space
        if distance <= self.social_space:
            # Calculate if paths might intersect
            human_speed = math.sqrt(human_vel.linear.x**2 + human_vel.linear.y**2)
            if human_speed > 0.1:  # Human is moving
                # Calculate time to closest approach
                relative_pos = [dx, dy]
                relative_vel = [
                    human_vel.linear.x,
                    human_vel.linear.y
                ]

                # Simple check: if human is approaching
                approach_rate = self.dot_product(relative_pos, relative_vel)

                if approach_rate < 0:  # Getting closer
                    return True

        return False
```

## Navigation Performance Optimization

### Efficient Path Planning and Execution

```python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import time
import threading

class OptimizedNavigation(Node):
    def __init__(self):
        super().__init__('optimized_navigation')

        # Performance monitoring
        self.planning_times = []
        self.execution_times = []
        self.lock = threading.Lock()

        # Publishers for performance metrics
        self.planning_time_pub = self.create_publisher(Float32, '/planning_time', 10)
        self.execution_time_pub = self.create_publisher(Float32, '/execution_time', 10)

        # Optimization parameters
        self.path_cache = {}  # Cache previously computed paths
        self.reuse_threshold = 0.5  # Reuse path if goal is close to previous

        # Threading for background processing
        self.planning_thread = None
        self.planning_in_progress = False

        # Pre-computed path segments for common locations
        self.precomputed_paths = {}

    def plan_path_optimized(self, start, goal, costmap):
        """Optimized path planning with caching"""
        start_time = time.time()

        # Create a key for path caching
        path_key = (round(start.position.x, 1), round(start.position.y, 1),
                   round(goal.position.x, 1), round(goal.position.y, 1))

        # Check if we have a cached path that's close enough
        if path_key in self.path_cache:
            cached_path, cache_time = self.path_cache[path_key]
            if time.time() - cache_time < 30:  # Cache valid for 30 seconds
                planning_time = time.time() - start_time
                self.publish_performance_metrics(planning_time, 0)
                return cached_path

        # Plan new path
        planner = GlobalPlanner()
        new_path = planner.plan(start, goal, costmap)

        # Cache the new path
        if new_path:
            self.path_cache[path_key] = (new_path, time.time())

        planning_time = time.time() - start_time
        self.publish_performance_metrics(planning_time, 0)

        return new_path

    def execute_path_optimized(self, path, current_pose, costmap):
        """Optimized path execution with dynamic replanning"""
        start_time = time.time()

        # Use local planner for execution
        local_planner = LocalPlanner()
        local_cmd = local_planner.plan(current_pose, path, costmap)

        execution_time = time.time() - start_time
        self.publish_performance_metrics(0, execution_time)

        return local_cmd

    def publish_performance_metrics(self, planning_time, execution_time):
        """Publish performance metrics"""
        with self.lock:
            if planning_time > 0:
                planning_msg = Float32()
                planning_msg.data = planning_time
                self.planning_time_pub.publish(planning_msg)
                self.planning_times.append(planning_time)

            if execution_time > 0:
                execution_msg = Float32()
                execution_msg.data = execution_time
                self.execution_time_pub.publish(execution_msg)
                self.execution_times.append(execution_time)

    def get_performance_stats(self):
        """Get performance statistics"""
        with self.lock:
            stats = {}
            if self.planning_times:
                stats['avg_planning_time'] = sum(self.planning_times) / len(self.planning_times)
                stats['max_planning_time'] = max(self.planning_times)
            if self.execution_times:
                stats['avg_execution_time'] = sum(self.execution_times) / len(self.execution_times)
                stats['max_execution_time'] = max(self.execution_times)

            return stats

def main(args=None):
    rclpy.init(args=args)
    nav_node = OptimizedNavigation()
    rclpy.spin(nav_node)
    nav_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Humanoid Navigation

### 1. Safety and Robustness
- Implement proper obstacle detection and avoidance
- Use multiple sensors for redundancy
- Validate navigation commands before execution
- Implement emergency stop mechanisms

### 2. Human-Centric Design
- Respect personal space of humans
- Consider social navigation norms
- Provide clear intentions to humans
- Use appropriate speeds for human environments

### 3. Performance Optimization
- Cache frequently used paths
- Use hierarchical planning (global + local)
- Optimize costmap updates
- Implement multi-threading for real-time performance

### 4. Adaptability
- Adjust parameters based on environment
- Learn from navigation experiences
- Handle dynamic environments
- Integrate with high-level task planning

## Summary

This chapter covered navigation and path planning for humanoid robots, including the Isaac ROS navigation system, costmap management, path planning algorithms, dynamic obstacle avoidance, and social navigation considerations. Humanoid navigation requires special attention to balance, human environments, and social interactions. The implementation includes global and local planners optimized for humanoid locomotion characteristics.

## Exercises

1. Implement a complete navigation stack for a humanoid robot simulation
2. Create a dynamic obstacle avoidance system using DWA
3. Develop social navigation behaviors for human-aware navigation
4. Optimize path planning algorithms for humanoid-specific constraints
5. Implement performance monitoring and optimization techniques

## Further Reading

- "Principles of Robot Motion" by Howie Choset et al.
- "Springer Handbook of Robotics" - Navigation chapter
- Social Force Model for pedestrian dynamics
- ROS Navigation Tutorials

---

*Next: [Chapter 5: Reinforcement Learning & Sim-to-Real Transfer](../chapter5-reinforcement-learning/README.md)*