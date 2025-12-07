# Chapter 3: Isaac ROS Perception Pipelines

## Learning Objectives

After completing this chapter, you will be able to:
- Design and implement perception pipelines using Isaac ROS packages
- Integrate multiple sensors for robust humanoid perception
- Configure Isaac ROS for object detection and tracking
- Implement SLAM systems for humanoid navigation
- Optimize perception pipelines for real-time performance

## Introduction to Isaac ROS Perception

Isaac ROS provides a comprehensive suite of perception packages specifically designed for robotics applications, with GPU acceleration for high-performance processing. For humanoid robots, perception systems must handle diverse sensor inputs including cameras, LIDAR, IMUs, and other modalities to enable navigation, manipulation, and interaction in human environments.

### Key Isaac ROS Perception Packages

1. **Isaac ROS Apriltag**: High-accuracy fiducial marker detection
2. **Isaac ROS DNN Inference**: GPU-accelerated neural network inference
3. **Isaac ROS Stereo DNN**: Stereo vision with deep learning
4. **Isaac ROS Visual SLAM**: Visual SLAM for localization and mapping
5. **Isaac ROS Segmentation**: Semantic and instance segmentation
6. **Isaac ROS Point Cloud**: Point cloud processing and fusion

## Isaac ROS Apriltag Detection

### Overview

Apriltag detection is crucial for humanoid robots for precise localization, calibration, and object identification. Isaac ROS provides high-performance Apriltag detection leveraging GPU acceleration.

### Installation and Configuration

```bash
# Install Isaac ROS Apriltag package
sudo apt install ros-humble-isaac-ros-apriltag

# Verify installation
ros2 pkg executables isaac_ros_apriltag
```

### Basic Apriltag Node Implementation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from isaac_ros_apriltag_interfaces.msg import AprilTagDetectionArray, AprilTagDetection
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np

class IsaacApriltagDetector(Node):
    def __init__(self):
        super().__init__('isaac_apriltag_detector')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Camera parameters (will be updated from camera_info)
        self.camera_matrix = None
        self.dist_coeffs = None

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_rect_color',
            self.image_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera_info',
            self.camera_info_callback,
            10
        )

        # Publisher for detections
        self.detection_pub = self.create_publisher(
            AprilTagDetectionArray,
            '/apriltag_detections',
            10
        )

        # Publisher for poses
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/apriltag_pose',
            10
        )

        # Parameters
        self.declare_parameter('family', 'tag36h11')
        self.declare_parameter('size', 0.14)  # Tag size in meters
        self.declare_parameter('max_hamming', 0)
        self.declare_parameter('quad_decimate', 2.0)
        self.declare_parameter('quad_sigma', 0.0)
        self.declare_parameter('refine_edges', 1)
        self.declare_parameter('decode_sharpening', 0.25)
        self.declare_parameter('debug', 0)

        self.family = self.get_parameter('family').value
        self.tag_size = self.get_parameter('size').value

        self.get_logger().info('Isaac Apriltag Detector initialized')

    def camera_info_callback(self, msg):
        """Update camera parameters from camera info"""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process incoming images for Apriltag detection"""
        if self.camera_matrix is None:
            self.get_logger().warn('Camera parameters not received yet')
            return

        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Run Apriltag detection
            detections = self.detect_apriltags(cv_image)

            # Publish detections
            self.publish_detections(detections, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def detect_apriltags(self, image):
        """Detect Apriltags in the image using GPU-accelerated method"""
        # In practice, this would use Isaac ROS's GPU-accelerated Apriltag detector
        # For this example, we'll use the CPU version for demonstration
        import apriltag

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Create detector
        detector = apriltag.Detector(
            families=self.family,
            nthreads=4,
            quad_decimate=self.get_parameter('quad_decimate').value,
            quad_sigma=self.get_parameter('quad_sigma').value,
            refine_edges=self.get_parameter('refine_edges').value,
            decode_sharpening=self.get_parameter('decode_sharpening').value,
            debug=self.get_parameter('debug').value
        )

        # Detect tags
        results = detector.detect(gray)

        return results

    def publish_detections(self, detections, header):
        """Publish Apriltag detections"""
        detection_array = AprilTagDetectionArray()
        detection_array.header = header

        for detection in detections:
            tag_detection = AprilTagDetection()
            tag_detection.id = [int(detection.tag_id)]
            tag_detection.size = [self.tag_size]

            # Convert corner points
            for corner in detection.corners:
                tag_detection.centre = detection.center.astype(np.float32)
                # Additional processing for pose estimation would go here

            detection_array.detections.append(tag_detection)

        self.detection_pub.publish(detection_array)

        # If we have detections, publish the first one's pose
        if detections:
            self.publish_pose(detections[0], header)

    def publish_pose(self, detection, header):
        """Publish the pose of the detected tag"""
        pose_msg = PoseStamped()
        pose_msg.header = header

        # Calculate pose from detection (simplified)
        # In practice, you'd use camera matrix and tag size for proper pose estimation
        pose_msg.pose.position.x = float(detection.center[0])
        pose_msg.pose.position.y = float(detection.center[1])
        pose_msg.pose.position.z = 0.0  # Simplified

        # Set orientation (identity for now)
        pose_msg.pose.orientation.w = 1.0

        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    detector = IsaacApriltagDetector()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Launch File for Apriltag Pipeline

```xml
<!-- launch/apriltag_pipeline.launch.py -->
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('my_robot_perception'),
        'config',
        'apriltag_config.yaml'
    )

    apriltag_node = Node(
        package='isaac_ros_apriltag',
        executable='apriltag_node',
        name='apriltag_node',
        parameters=[
            config,
            {
                'family': 'tag36h11',
                'size': 0.14,
                'max_hamming': 0,
                'quad_decimate': 2.0,
                'quad_sigma': 0.0,
                'refine_edges': 1,
                'decode_sharpening': 0.25,
                'debug': 0
            }
        ],
        remappings=[
            ('image', '/camera/image_rect_color'),
            ('camera_info', '/camera/camera_info'),
            ('detections', '/apriltag_detections')
        ]
    )

    return LaunchDescription([
        apriltag_node
    ])
```

## Isaac ROS DNN Inference for Humanoid Perception

### Overview

Deep neural network inference is essential for humanoid robots to perform object detection, semantic segmentation, and other AI tasks. Isaac ROS provides GPU-accelerated DNN inference capabilities.

### DNN Inference Node Implementation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import torch
import torchvision.transforms as transforms

class IsaacDNNInference(Node):
    def __init__(self):
        super().__init__('isaac_dnn_inference')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Initialize model (placeholder - would load actual model)
        self.model = self.load_model()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((416, 416)),  # YOLO input size
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_rect_color',
            self.image_callback,
            10
        )

        # Publisher for detections
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            '/dnn_detections',
            10
        )

        # Configuration parameters
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

        self.get_logger().info('Isaac DNN Inference node initialized')

    def load_model(self):
        """Load pre-trained neural network model"""
        # In practice, you would load a TensorRT optimized model
        # or a model specifically designed for Isaac ROS
        try:
            # Placeholder model loading
            # This would typically load a model optimized for TensorRT
            import torch
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.eval()
            return model
        except Exception as e:
            self.get_logger().error(f'Error loading model: {e}')
            return None

    def image_callback(self, msg):
        """Process incoming image for DNN inference"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Run inference
            detections = self.run_inference(cv_image)

            # Publish results
            self.publish_detections(detections, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')

    def run_inference(self, image):
        """Run DNN inference on the image"""
        if self.model is None:
            return []

        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image)

            # Run inference
            with torch.no_grad():
                results = self.model(input_tensor)

            # Process results
            detections = self.process_results(results, image.shape)

            return detections

        except Exception as e:
            self.get_logger().error(f'Error during inference: {e}')
            return []

    def preprocess_image(self, image):
        """Preprocess image for neural network input"""
        # Resize image to model input size
        img_resized = cv2.resize(image, (416, 416))

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize
        input_tensor = self.transform(img_rgb).unsqueeze(0)  # Add batch dimension

        return input_tensor

    def process_results(self, results, original_shape):
        """Process neural network results into Detection2DArray format"""
        detections = []

        # Extract detections from results (this depends on the model)
        # For YOLOv5, results.pred contains the detections
        if hasattr(results, 'xyxy'):
            # YOLO format: [x1, y1, x2, y2, conf, class]
            for detection in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = detection

                if conf >= self.confidence_threshold:
                    # Convert to vision_msgs format
                    det = Detection2D()
                    det.header = Header()
                    det.results = []

                    # Set bounding box
                    bbox = Point()
                    bbox.x = float(x1)
                    bbox.y = float(y1)
                    bbox.z = 0.0

                    det.bbox.center = bbox
                    det.bbox.size_x = float(x2 - x1)
                    det.bbox.size_y = float(y2 - y1)

                    # Set confidence
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = str(int(cls))
                    hypothesis.hypothesis.score = float(conf)

                    det.results.append(hypothesis)
                    detections.append(det)

        return detections

    def publish_detections(self, detections, header):
        """Publish detection results"""
        detection_array = Detection2DArray()
        detection_array.header = header
        detection_array.detections = detections

        self.detection_pub.publish(detection_array)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacDNNInference()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Isaac ROS Stereo DNN Implementation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from stereo_msgs.msg import DisparityImage
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import struct

class IsaacStereoDNN(Node):
    def __init__(self):
        super().__init__('isaac_stereo_dnn')

        # Camera parameters
        self.left_cam_info = None
        self.right_cam_info = None
        self.Q_matrix = None

        # Subscriptions
        self.left_image_sub = self.create_subscription(
            Image, '/stereo/left/image_rect_color', self.left_image_callback, 10
        )
        self.right_image_sub = self.create_subscription(
            Image, '/stereo/right/image_rect_color', self.right_image_callback, 10
        )
        self.left_cam_info_sub = self.create_subscription(
            CameraInfo, '/stereo/left/camera_info', self.left_cam_info_callback, 10
        )
        self.right_cam_info_sub = self.create_subscription(
            CameraInfo, '/stereo/right/camera_info', self.right_cam_info_callback, 10
        )

        # Publishers
        self.disparity_pub = self.create_publisher(DisparityImage, '/disparity_map', 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/stereo_pointcloud', 10)

        # Store images
        self.latest_left = None
        self.latest_right = None

        self.get_logger().info('Isaac Stereo DNN node initialized')

    def left_cam_info_callback(self, msg):
        """Handle left camera info"""
        self.left_cam_info = msg
        self.compute_stereo_params()

    def right_cam_info_callback(self, msg):
        """Handle right camera info"""
        self.right_cam_info = msg
        self.compute_stereo_params()

    def compute_stereo_params(self):
        """Compute stereo rectification parameters"""
        if self.left_cam_info and self.right_cam_info:
            # Extract Q matrix from stereo parameters
            # This is a simplified version - in practice you'd use proper stereo calibration
            self.Q_matrix = self.compute_disparity_to_depth_matrix()

    def compute_disparity_to_depth_matrix(self):
        """Compute the Q matrix for disparity to depth conversion"""
        # Simplified Q matrix computation
        # In practice, this comes from stereo calibration
        Tx = -0.1  # Baseline (example: 10cm)
        f = 600.0  # Focal length (example)

        Q = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, f],
            [0, 0, -1/Tx, 0]
        ], dtype=np.float32)

        return Q

    def left_image_callback(self, msg):
        """Process left stereo image"""
        self.latest_left = msg
        self.process_stereo_pair()

    def right_image_callback(self, msg):
        """Process right stereo image"""
        self.latest_right = msg
        self.process_stereo_pair()

    def process_stereo_pair(self):
        """Process stereo image pair to generate disparity and point cloud"""
        if self.latest_left is None or self.latest_right is None:
            return

        # In Isaac ROS, stereo processing is typically done with specialized nodes
        # Here we'll simulate the process

        # Convert images to OpenCV
        from cv_bridge import CvBridge
        bridge = CvBridge()

        try:
            left_cv = bridge.imgmsg_to_cv2(self.latest_left, "bgr8")
            right_cv = bridge.imgmsg_to_cv2(self.latest_right, "bgr8")

            # Compute disparity (simplified - in practice use Isaac ROS stereo node)
            disparity = self.compute_disparity(left_cv, right_cv)

            # Create and publish disparity image
            self.publish_disparity(disparity, self.latest_left.header)

            # Generate point cloud from disparity
            if self.Q_matrix is not None:
                pointcloud = self.disparity_to_pointcloud(disparity)
                self.publish_pointcloud(pointcloud, self.latest_left.header)

        except Exception as e:
            self.get_logger().error(f'Error processing stereo pair: {e}')

    def compute_disparity(self, left_img, right_img):
        """Compute disparity map from stereo images"""
        # In Isaac ROS, this would use GPU-accelerated stereo matching
        # For this example, we'll use OpenCV's StereoSGBM
        import cv2

        # Convert to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

        # Create stereo matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        return disparity

    def publish_disparity(self, disparity, header):
        """Publish disparity image"""
        from cv_bridge import CvBridge
        bridge = CvBridge()

        # Create disparity message
        disp_msg = DisparityImage()
        disp_msg.header = header
        disp_msg.image = bridge.cv2_to_imgmsg(disparity, encoding="32FC1")
        disp_msg.f = 600.0  # Focal length
        disp_msg.T = 0.1    # Baseline
        disp_msg.min_disparity = 0.0
        disp_msg.max_disparity = 64.0
        disp_msg.delta_d = 0.1666666

        self.disparity_pub.publish(disp_msg)

    def disparity_to_pointcloud(self, disparity):
        """Convert disparity map to point cloud"""
        if self.Q_matrix is None:
            return np.array([])

        # Get valid disparity points
        valid = disparity > 0
        if not np.any(valid):
            return np.array([])

        # Get image coordinates
        height, width = disparity.shape
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        y_coords = y_coords[valid]
        x_coords = x_coords[valid]
        disp_values = disparity[valid]

        # Create homogeneous coordinates
        ones = np.ones_like(x_coords)
        img_coords = np.stack([x_coords, y_coords, disp_values, ones], axis=1)

        # Apply Q matrix transformation
        points = img_coords @ self.Q_matrix.T

        # Convert from homogeneous to Cartesian coordinates
        points = points[:, :3] / points[:, 3:4]

        return points

    def publish_pointcloud(self, points, header):
        """Publish point cloud"""
        if len(points) == 0:
            return

        # Create PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.is_dense = False
        cloud_msg.is_bigendian = False

        # Define fields
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
        ]
        cloud_msg.fields = fields
        cloud_msg.point_step = 12  # 3 * 4 bytes
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width

        # Pack points into binary data
        data = []
        for point in points:
            data.append(struct.pack('fff', point[0], point[1], point[2]))

        cloud_msg.data = b''.join(data)

        self.pointcloud_pub.publish(cloud_msg)

def main(args=None):
    rclpy.init(args=args)
    node = IsaacStereoDNN()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Isaac ROS Visual SLAM for Humanoid Navigation

### Overview of Visual SLAM

Visual SLAM (Simultaneous Localization and Mapping) is crucial for humanoid robots to navigate unknown environments. Isaac ROS provides GPU-accelerated Visual SLAM capabilities.

### Visual SLAM Node Implementation

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np

class IsaacVisualSLAM(Node):
    def __init__(self):
        super().__init__('isaac_visual_slam')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # SLAM state
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.keyframes = []
        self.map_points = []

        # Subscriptions
        self.image_sub = self.create_subscription(
            Image, '/camera/image_rect_color', self.image_callback, 10
        )
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/visual_odom', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/visual_pose', 10)

        # Feature tracking parameters
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        self.descriptor_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Previous frame data
        self.prev_image = None
        self.prev_kp = None
        self.prev_desc = None

        self.get_logger().info('Isaac Visual SLAM node initialized')

    def image_callback(self, msg):
        """Process incoming images for SLAM"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Process frame for SLAM
            self.process_slam_frame(cv_image, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def imu_callback(self, msg):
        """Process IMU data for sensor fusion"""
        # In Isaac ROS Visual SLAM, IMU data would be used for
        # motion prediction and sensor fusion
        # For this example, we'll just store the data
        self.latest_imu = msg

    def process_slam_frame(self, image, header):
        """Process a single frame for SLAM"""
        # Extract features from current image
        kp, desc = self.extract_features(image)

        if self.prev_kp is not None and self.prev_desc is not None:
            # Match features with previous frame
            matches = self.match_features(self.prev_desc, desc)

            if len(matches) >= 10:  # Need sufficient matches
                # Estimate motion between frames
                motion = self.estimate_motion(
                    self.prev_kp, kp, matches
                )

                # Update pose
                self.update_pose(motion)

                # Publish pose and odometry
                self.publish_pose_estimate(header)
                self.publish_odometry(header)

        # Update previous frame data
        self.prev_image = image
        self.prev_kp = kp
        self.prev_desc = desc

    def extract_features(self, image):
        """Extract features from image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect and compute descriptors
        kp = self.feature_detector.detect(gray)
        kp, desc = self.feature_detector.compute(gray, kp)

        if desc is not None:
            # Convert keypoints to numpy array
            kp_array = np.array([k.pt for k in kp])
        else:
            kp_array = np.array([])
            desc = np.array([])

        return kp_array, desc

    def match_features(self, desc1, desc2):
        """Match features between two frames"""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        try:
            # Use FLANN matcher for better performance
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)

            matches = flann.knnMatch(desc1, desc2, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

            return good_matches

        except Exception as e:
            self.get_logger().warn(f'Feature matching error: {e}')
            return []

    def estimate_motion(self, prev_kp, curr_kp, matches):
        """Estimate motion between frames using feature matches"""
        if len(matches) < 10:
            return np.eye(4)

        # Get matched points
        prev_pts = np.float32([prev_kp[m.queryIdx] for m in matches]).reshape(-1, 1, 2)
        curr_pts = np.float32([curr_kp[m.trainIdx] for m in matches]).reshape(-1, 1, 2)

        # Estimate essential matrix
        E, mask = cv2.findEssentialMat(
            curr_pts, prev_pts,
            focal=600,  # Camera focal length
            pp=(320, 240),  # Principal point
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is not None:
            # Recover pose
            _, R, t, _ = cv2.recoverPose(E, curr_pts, prev_pts)

            # Create transformation matrix
            motion = np.eye(4)
            motion[:3, :3] = R
            motion[:3, 3] = t.ravel()

            return motion

        return np.eye(4)

    def update_pose(self, motion):
        """Update current pose with estimated motion"""
        self.current_pose = self.current_pose @ np.linalg.inv(motion)

    def publish_pose_estimate(self, header):
        """Publish current pose estimate"""
        pose_msg = PoseStamped()
        pose_msg.header = header

        # Convert transformation matrix to pose
        pos = self.current_pose[:3, 3]
        pose_msg.pose.position.x = float(pos[0])
        pose_msg.pose.position.y = float(pos[1])
        pose_msg.pose.position.z = float(pos[2])

        # Convert rotation matrix to quaternion
        R = self.current_pose[:3, :3]
        qw, qx, qy, qz = self.rotation_matrix_to_quaternion(R)
        pose_msg.pose.orientation.w = qw
        pose_msg.pose.orientation.x = qx
        pose_msg.pose.orientation.y = qy
        pose_msg.pose.orientation.z = qz

        self.pose_pub.publish(pose_msg)

    def publish_odometry(self, header):
        """Publish odometry message"""
        odom_msg = Odometry()
        odom_msg.header = header
        odom_msg.child_frame_id = "base_link"

        # Set pose
        pos = self.current_pose[:3, 3]
        odom_msg.pose.pose.position.x = float(pos[0])
        odom_msg.pose.pose.position.y = float(pos[1])
        odom_msg.pose.pose.position.z = float(pos[2])

        # Set orientation
        R = self.current_pose[:3, :3]
        qw, qx, qy, qz = self.rotation_matrix_to_quaternion(R)
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.pose.pose.orientation.x = qx
        odom_msg.pose.pose.orientation.y = qy
        odom_msg.pose.pose.orientation.z = qz

        # Set covariances (simplified)
        odom_msg.pose.covariance = [0.0] * 36  # Placeholder
        odom_msg.twist.covariance = [0.0] * 36  # Placeholder

        self.odom_pub.publish(odom_msg)

    def rotation_matrix_to_quaternion(self, R):
        """Convert 3x3 rotation matrix to quaternion"""
        # Method from: https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
        trace = np.trace(R)

        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2  # S=4*qw
            qw = 0.25 * s
            qx = (R[2, 1] - R[1, 2]) / s
            qy = (R[0, 2] - R[2, 0]) / s
            qz = (R[1, 0] - R[0, 1]) / s
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # S=4*qx
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # S=4*qy
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # S=4*qz
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

        # Normalize quaternion
        norm = np.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
        return qw/norm, qx/norm, qy/norm, qz/norm

def main(args=None):
    rclpy.init(args=args)
    node = IsaacVisualSLAM()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Multi-Sensor Fusion for Humanoid Perception

### Sensor Fusion Architecture

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, LaserScan, PointCloud2
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
import numpy as np
from scipy.spatial.transform import Rotation as R

class HumanoidSensorFusion(Node):
    def __init__(self):
        super().__init__('humanoid_sensor_fusion')

        # Initialize internal state
        self.robot_pose = np.zeros(6)  # [x, y, z, roll, pitch, yaw]
        self.robot_velocity = np.zeros(6)  # [vx, vy, vz, wx, wy, wz]
        self.pose_covariance = np.eye(6) * 0.1  # Initial uncertainty
        self.velocity_covariance = np.eye(6) * 0.1

        # Time tracking
        self.last_update_time = self.get_clock().now()

        # Subscriptions for different sensors
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_callback, 10
        )
        self.odom_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/visual_odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )

        # Publisher for fused state
        self.fused_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/fused_pose', 10
        )
        self.fused_velocity_pub = self.create_publisher(
            TwistWithCovarianceStamped, '/fused_velocity', 10
        )

        # Transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Sensor data storage
        self.imu_data = None
        self.odom_data = None
        self.scan_data = None

        # Kalman filter components
        self.process_noise = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01])  # Process noise
        self.imu_noise = np.diag([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])  # IMU noise
        self.odom_noise = np.diag([0.05, 0.05, 0.05, 0.01, 0.01, 0.01])  # Odometry noise

        # Setup timer for fusion loop
        self.fusion_timer = self.create_timer(0.01, self.fusion_loop)  # 100Hz

        self.get_logger().info('Humanoid Sensor Fusion node initialized')

    def imu_callback(self, msg):
        """Handle IMU data"""
        # Extract acceleration and angular velocity
        linear_acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        angular_vel = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Store IMU data with timestamp
        self.imu_data = {
            'linear_acc': linear_acc,
            'angular_vel': angular_vel,
            'orientation': [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w
            ],
            'timestamp': msg.header.stamp
        }

    def odom_callback(self, msg):
        """Handle odometry data"""
        # Extract position and orientation
        position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        orientation = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])

        # Store odometry data
        self.odom_data = {
            'position': position,
            'orientation': orientation,
            'timestamp': msg.header.stamp,
            'pose_covariance': np.array(msg.pose.covariance).reshape(6, 6)
        }

    def scan_callback(self, msg):
        """Handle laser scan data"""
        # Process laser scan for environment understanding
        # This could be used for additional localization constraints
        self.scan_data = {
            'ranges': np.array(msg.ranges),
            'intensities': np.array(msg.intensities),
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': msg.angle_increment,
            'timestamp': msg.header.stamp
        }

    def fusion_loop(self):
        """Main fusion loop running at 100Hz"""
        current_time = self.get_clock().now()
        dt = (current_time - self.last_update_time).nanoseconds / 1e9
        self.last_update_time = current_time

        if dt <= 0:
            return

        # Prediction step - predict state forward in time
        self.predict_state(dt)

        # Update step - incorporate sensor measurements
        if self.imu_data is not None:
            self.update_with_imu()

        if self.odom_data is not None:
            self.update_with_odom()

        # Publish fused state
        self.publish_fused_state(current_time.to_msg())

        # Broadcast transform
        self.broadcast_transform(current_time.to_msg())

    def predict_state(self, dt):
        """Predict state forward using IMU data"""
        if self.imu_data is None:
            return

        # Extract IMU measurements
        linear_acc = self.imu_data['linear_acc']
        angular_vel = self.imu_data['angular_vel']

        # Get current orientation to transform acceleration to world frame
        q = self.imu_data['orientation']
        rotation = R.from_quat(q)

        # Transform acceleration from body frame to world frame
        world_acc = rotation.apply(linear_acc)

        # Update position (integrate velocity)
        self.robot_pose[:3] += self.robot_velocity[:3] * dt + 0.5 * world_acc * dt**2

        # Update orientation (integrate angular velocity)
        # This is a simplified approach - in practice, use quaternion integration
        angular_vel_world = rotation.apply(angular_vel)
        self.robot_pose[3:] += angular_vel_world * dt

        # Normalize angles to [-π, π]
        self.robot_pose[3:] = np.arctan2(
            np.sin(self.robot_pose[3:]),
            np.cos(self.robot_pose[3:])
        )

        # Update velocity (integrate acceleration)
        self.robot_velocity[:3] += world_acc * dt

        # Update covariance (simplified model)
        F = self.get_jacobian_F(dt)  # State transition Jacobian
        self.pose_covariance = F @ self.pose_covariance @ F.T + self.process_noise * dt

    def get_jacobian_F(self, dt):
        """Get state transition Jacobian for covariance prediction"""
        F = np.eye(6)
        # For a constant velocity model:
        # position = position + velocity * dt
        F[0:3, 3:6] = np.eye(3) * dt
        return F

    def update_with_imu(self):
        """Update state estimate with IMU measurement"""
        # IMU provides orientation and angular velocity
        # Update orientation part of state
        orientation_measurement = self.imu_data['orientation']

        # Convert to roll, pitch, yaw for comparison
        meas_rotation = R.from_quat(orientation_measurement)
        meas_rpy = meas_rotation.as_euler('xyz')

        # Innovation (difference between measurement and prediction)
        innovation = self.angle_difference(meas_rpy, self.robot_pose[3:])

        # Simplified update (in practice, use proper EKF equations)
        # For now, we'll use a simple complementary filter approach
        alpha = 0.1  # Filter gain for orientation
        self.robot_pose[3:] = self.robot_pose[3:] + alpha * innovation

    def update_with_odom(self):
        """Update state estimate with odometry measurement"""
        # Extract measurement and its covariance
        meas_pos = self.odom_data['position']
        meas_rot = R.from_quat(self.odom_data['orientation'])
        meas_rpy = meas_rot.as_euler('xyz')

        measurement = np.concatenate([meas_pos, meas_rpy])
        meas_cov = self.odom_data['pose_covariance']

        # Innovation
        innovation = self.angle_difference(measurement, self.robot_pose)

        # Simplified Kalman update
        # In practice, compute Kalman gain properly
        K_pos = 0.8  # Position update gain
        K_rot = 0.3  # Orientation update gain

        # Update state
        self.robot_pose[:3] += K_pos * innovation[:3]
        self.robot_pose[3:] += K_rot * innovation[3:]

        # Update covariance (simplified)
        I_KH = np.eye(6) - np.eye(6) * np.array([K_pos, K_pos, K_pos, K_rot, K_rot, K_rot])
        self.pose_covariance = I_KH @ self.pose_covariance

    def angle_difference(self, angle1, angle2):
        """Compute angle difference accounting for angle wrapping"""
        diff = angle1 - angle2
        # Wrap angles to [-π, π]
        diff = np.arctan2(np.sin(diff), np.cos(diff))
        return diff

    def publish_fused_state(self, header):
        """Publish the fused state estimate"""
        # Publish pose
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = header
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.pose.position.x = float(self.robot_pose[0])
        pose_msg.pose.position.y = float(self.robot_pose[1])
        pose_msg.pose.position.z = float(self.robot_pose[2])

        # Convert orientation to quaternion
        orientation_r = R.from_euler('xyz', self.robot_pose[3:])
        quat = orientation_r.as_quat()
        pose_msg.pose.pose.orientation.x = float(quat[0])
        pose_msg.pose.pose.orientation.y = float(quat[1])
        pose_msg.pose.pose.orientation.z = float(quat[2])
        pose_msg.pose.pose.orientation.w = float(quat[3])

        # Set covariance
        pose_msg.pose.covariance = self.pose_covariance.flatten().tolist()

        self.fused_pose_pub.publish(pose_msg)

        # Publish velocity
        vel_msg = TwistWithCovarianceStamped()
        vel_msg.header.stamp = header
        vel_msg.header.frame_id = 'base_link'

        vel_msg.twist.twist.linear.x = float(self.robot_velocity[0])
        vel_msg.twist.twist.linear.y = float(self.robot_velocity[1])
        vel_msg.twist.twist.linear.z = float(self.robot_velocity[2])
        vel_msg.twist.twist.angular.x = float(self.robot_velocity[3])
        vel_msg.twist.twist.angular.y = float(self.robot_velocity[4])
        vel_msg.twist.twist.angular.z = float(self.robot_velocity[5])

        vel_msg.twist.covariance = self.velocity_covariance.flatten().tolist()

        self.fused_velocity_pub.publish(vel_msg)

    def broadcast_transform(self, header):
        """Broadcast transform from map to base_link"""
        t = TransformStamped()

        t.header.stamp = header
        t.header.frame_id = 'map'
        t.child_frame_id = 'base_link'

        t.transform.translation.x = float(self.robot_pose[0])
        t.transform.translation.y = float(self.robot_pose[1])
        t.transform.translation.z = float(self.robot_pose[2])

        orientation_r = R.from_euler('xyz', self.robot_pose[3:])
        quat = orientation_r.as_quat()
        t.transform.rotation.x = float(quat[0])
        t.transform.rotation.y = float(quat[1])
        t.transform.rotation.z = float(quat[2])
        t.transform.rotation.w = float(quat[3])

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidSensorFusion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Performance Optimization for Perception Pipelines

### Pipeline Optimization Techniques

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from threading import Lock
import numpy as np
import time

class OptimizedPerceptionPipeline(Node):
    def __init__(self):
        super().__init__('optimized_perception_pipeline')

        # Performance monitoring
        self.frame_times = []
        self.lock = Lock()

        # QoS for performance
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
        qos_profile = QoSProfile(
            depth=1,  # Minimal queue to reduce latency
            reliability=QoSReliabilityPolicy.BEST_EFFORT,  # For performance
            history=QoSHistoryPolicy.KEEP_LAST
        )

        # Subscription with optimized QoS
        self.image_sub = self.create_subscription(
            Image, '/camera/image_rect_color', self.optimized_image_callback, qos_profile
        )

        # Publisher for performance metrics
        self.perf_pub = self.create_publisher(Int32, '/perception_performance', 10)

        # Pre-allocated arrays for performance
        self.processed_frame = None
        self.detection_results = []

        # Processing pipeline steps
        self.enable_preprocessing = True
        self.enable_detection = True
        self.enable_postprocessing = True

        # Performance thresholds
        self.target_fps = 30
        self.max_processing_time = 1.0 / self.target_fps  # seconds

        self.get_logger().info('Optimized Perception Pipeline initialized')

    def optimized_image_callback(self, msg):
        """Optimized image callback with performance considerations"""
        start_time = time.time()

        try:
            # Convert image with minimal overhead
            cv_image = self.convert_image_optimized(msg)

            # Process pipeline with optimization
            if self.enable_preprocessing:
                cv_image = self.preprocess_image(cv_image)

            if self.enable_detection:
                results = self.run_detection_optimized(cv_image)

            if self.enable_postprocessing:
                results = self.postprocess_results(results)

            # Update performance metrics
            processing_time = time.time() - start_time

            # Store performance data thread-safely
            with self.lock:
                self.frame_times.append(processing_time)

                # Keep only last 100 frames for performance calculation
                if len(self.frame_times) > 100:
                    self.frame_times.pop(0)

            # Publish performance metric
            perf_msg = Int32()
            perf_msg.data = int(1.0 / processing_time) if processing_time > 0 else 0
            self.perf_pub.publish(perf_msg)

        except Exception as e:
            self.get_logger().error(f'Error in optimized callback: {e}')

    def convert_image_optimized(self, ros_msg):
        """Optimized image conversion"""
        # In practice, use cv_bridge with pre-allocated memory
        # or use Isaac ROS's optimized image transport
        pass

    def preprocess_image(self, image):
        """Optimized preprocessing pipeline"""
        # Resize only if necessary
        # Use GPU for heavy operations
        # Avoid unnecessary color space conversions
        pass

    def run_detection_optimized(self, image):
        """Optimized detection with GPU acceleration"""
        # Use TensorRT optimized models
        # Batch processing when possible
        # Asynchronous execution
        pass

    def postprocess_results(self, results):
        """Optimized postprocessing"""
        # Minimal data copying
        # Efficient data structures
        pass

    def get_performance_stats(self):
        """Get current performance statistics"""
        with self.lock:
            if not self.frame_times:
                return {'fps': 0, 'avg_time': 0, 'min_time': 0, 'max_time': 0}

            times = np.array(self.frame_times)
            fps_values = 1.0 / times

            return {
                'fps': float(np.mean(fps_values)),
                'avg_time': float(np.mean(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'latency_95th': float(np.percentile(times, 95))
            }

class PipelineScheduler(Node):
    def __init__(self):
        super().__init__('pipeline_scheduler')

        # Priority levels for different perception tasks
        self.task_priorities = {
            'obstacle_detection': 1,  # Highest priority
            'person_detection': 2,
            'object_recognition': 3,
            'environment_mapping': 4  # Lowest priority
        }

        # Task execution times (estimated)
        self.task_times = {
            'obstacle_detection': 0.02,   # 20ms
            'person_detection': 0.05,     # 50ms
            'object_recognition': 0.1,    # 100ms
            'environment_mapping': 0.2    # 200ms
        }

        # Task scheduling
        self.scheduled_tasks = []
        self.current_task = None

        # Setup timer for task scheduling
        self.schedule_timer = self.create_timer(0.01, self.schedule_tasks)  # 100Hz

    def schedule_tasks(self):
        """Schedule perception tasks based on priority and timing constraints"""
        # Implement priority-based scheduling
        # Consider deadline constraints
        # Balance between accuracy and performance
        pass

def main(args=None):
    rclpy.init(args=args)

    # Create both nodes
    pipeline = OptimizedPerceptionPipeline()
    scheduler = PipelineScheduler()

    # Spin both nodes
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(pipeline)
    executor.add_node(scheduler)

    try:
        executor.spin()
    finally:
        pipeline.destroy_node()
        scheduler.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Best Practices for Isaac ROS Perception

### 1. GPU Utilization
- Leverage Isaac ROS's GPU-accelerated packages
- Use TensorRT optimized models
- Batch processing when possible
- Optimize memory transfers between CPU and GPU

### 2. Sensor Integration
- Synchronize sensor timestamps properly
- Use hardware synchronization when available
- Implement proper calibration procedures
- Validate sensor data quality

### 3. Real-time Performance
- Use appropriate QoS settings for low latency
- Implement multi-threading for parallel processing
- Monitor and optimize processing pipelines
- Use performance profiling tools

### 4. Robustness
- Handle sensor failures gracefully
- Implement fallback mechanisms
- Validate sensor data before processing
- Use sensor fusion for redundancy

## Summary

This chapter covered Isaac ROS perception pipelines, including Apriltag detection, DNN inference, stereo vision, and Visual SLAM for humanoid robots. We explored multi-sensor fusion techniques and performance optimization strategies. Isaac ROS provides powerful GPU-accelerated perception capabilities essential for humanoid robots operating in complex environments.

## Exercises

1. Implement an Isaac ROS Apriltag detection pipeline for humanoid localization
2. Create a DNN inference node for object detection in humanoid environments
3. Build a stereo vision pipeline for depth estimation
4. Implement a Visual SLAM system for humanoid navigation
5. Design a multi-sensor fusion system combining camera, IMU, and LIDAR

## Further Reading

- NVIDIA Isaac ROS Documentation: https://nvidia-isaac-ros.github.io/
- "Probabilistic Robotics" by Thrun, Burgard, and Fox
- GPU-Accelerated Computer Vision with CUDA

---

*Next: [Chapter 4: Navigation & Path Planning](../chapter4-navigation/README.md)*