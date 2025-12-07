# Chapter 4: Unity for High-Fidelity Robot Visualization

## Learning Objectives

After completing this chapter, you will be able to:
- Set up Unity for robotics visualization and simulation
- Import and configure robot models in Unity
- Implement realistic physics and collision systems
- Create high-fidelity visualizations for humanoid robots
- Integrate Unity with ROS 2 for real-time control

## Introduction to Unity for Robotics

Unity is a powerful 3D development platform that provides high-fidelity visualization capabilities for robotics applications. Unlike traditional robotics simulators focused primarily on physics, Unity excels at creating photorealistic environments and visualizations that can be crucial for tasks like perception training, human-robot interaction studies, and presentation-quality demonstrations.

### Unity Robotics Features

- **High-Fidelity Graphics**: Photorealistic rendering with advanced lighting and materials
- **XR Support**: Virtual and augmented reality capabilities
- **Asset Store**: Extensive library of 3D models, environments, and tools
- **Cross-Platform Deployment**: Windows, Linux, macOS, and mobile platforms
- **Scripting**: C# scripting with extensive API access
- **Real-time Performance**: Optimized for real-time applications

### Unity Robotics Hub

Unity provides the Robotics Hub, a collection of tools and packages specifically designed for robotics applications:

- **Unity ML-Agents**: For reinforcement learning and AI training
- **ROS#**: For ROS/ROS 2 communication
- **Unity Perception**: For synthetic data generation
- **Omniverse Integration**: For advanced simulation capabilities

## Setting Up Unity for Robotics

### Prerequisites and Installation

1. Install Unity Hub from https://unity.com/
2. Install Unity 2021.3 LTS or later (recommended for stability)
3. Install required packages via Unity Package Manager

### Installing Robotics Packages

```csharp
// Example package installation via Package Manager
// In Unity Editor: Window > Package Manager
// Install the following packages:
// - ROS TCP Connector (for ROS communication)
// - Unity Perception (for synthetic data)
// - ML-Agents (for AI training)
```

### Unity ROS# Setup

Unity ROS# enables communication between Unity and ROS 2:

1. Import the ROS TCP Connector package
2. Configure the TCP connection settings
3. Set up message serialization

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Std;
using RosMessageTypes.Sensor;

public class UnityRobotController : MonoBehaviour
{
    ROSConnection ros;
    string ros_hostname = "127.0.0.1";
    int ros_port = 10000;

    // Robot joint variables
    public Transform headJoint;
    public Transform leftShoulder;
    public Transform rightShoulder;
    public Transform leftElbow;
    public Transform rightElbow;
    public Transform leftHip;
    public Transform rightHip;
    public Transform leftKnee;
    public Transform rightKnee;
    public Transform leftAnkle;
    public Transform rightAnkle;

    // Start is called before the first frame update
    void Start()
    {
        // Get the ROS connection static instance
        ros = ROSConnection.GetOrCreateInstance();
        ros.hostname = ros_hostname;
        ros.port = ros_port;

        // Start the ROS publisher
        StartCoroutine(ROSUpdate());
    }

    IEnumerator ROSUpdate()
    {
        // Subscribe to joint state messages
        ros.Subscribe<sensor_msgs.JointStateMsg>("joint_states", JointStateCallback);

        while (true)
        {
            // Publish robot state
            PublishRobotState();

            yield return new WaitForSeconds(0.01f); // 100Hz update
        }
    }

    void JointStateCallback(sensor_msgs.JointStateMsg jointState)
    {
        // Update robot visualization based on joint states
        for (int i = 0; i < jointState.name.Count; i++)
        {
            string jointName = jointState.name[i];
            float jointPosition = jointState.position[i];

            switch (jointName)
            {
                case "head_joint":
                    headJoint.localRotation = Quaternion.Euler(0, 0, jointPosition * Mathf.Rad2Deg);
                    break;
                case "left_shoulder_joint":
                    leftShoulder.localRotation = Quaternion.Euler(0, jointPosition * Mathf.Rad2Deg, 0);
                    break;
                case "right_shoulder_joint":
                    rightShoulder.localRotation = Quaternion.Euler(0, jointPosition * Mathf.Rad2Deg, 0);
                    break;
                // Add cases for other joints...
            }
        }
    }

    void PublishRobotState()
    {
        // Create and publish robot state message
        sensor_msgs.JointStateMsg state = new sensor_msgs.JointStateMsg();
        state.header = new std_msgs.HeaderMsg();
        state.header.stamp = new builtin_interfaces.TimeMsg(
            System.DateTime.UtcNow.Second,
            System.DateTime.UtcNow.Millisecond * 1000000);

        // Add joint names and positions
        state.name.Add("head_joint");
        state.position.Add(headJoint.localRotation.eulerAngles.z * Mathf.Deg2Rad);

        // Add other joints...

        // Publish the message
        ros.Publish("unity_joint_states", state);
    }
}
```

## Robot Model Import and Configuration

### Importing Robot Models

Unity supports various 3D model formats, but for robotics applications, consider:

1. **FBX**: Industry standard, supports animations and materials
2. **OBJ**: Simple format, good for basic geometry
3. **URDF Import**: Unity now has tools to import URDF directly

### Robot Configuration in Unity

```csharp
using UnityEngine;

public class RobotConfiguration : MonoBehaviour
{
    [Header("Robot Dimensions")]
    public float torsoHeight = 0.6f;
    public float torsoWidth = 0.25f;
    public float torsoDepth = 0.25f;

    [Header("Joint Limits")]
    [Range(-180f, 180f)] public float hipJointLimit = 45f;
    [Range(-180f, 180f)] public float kneeJointLimit = 90f;
    [Range(-180f, 180f)] public float ankleJointLimit = 30f;

    [Header("Physical Properties")]
    public float robotMass = 50f;
    public float centerOfMassHeight = 0.8f;

    // Joint configuration
    [System.Serializable]
    public class JointConfig
    {
        public string jointName;
        public Transform jointTransform;
        public JointType jointType;
        public Vector2 jointLimits; // min, max in degrees
        public float jointSpeed = 90f; // degrees per second
    }

    [Header("Joint Configurations")]
    public List<JointConfig> joints = new List<JointConfig>();

    void Start()
    {
        ConfigureRobot();
    }

    void ConfigureRobot()
    {
        // Set up physical properties
        ConfigurePhysicalProperties();

        // Configure each joint
        foreach (JointConfig joint in joints)
        {
            ConfigureJoint(joint);
        }
    }

    void ConfigurePhysicalProperties()
    {
        // Set center of mass
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.centerOfMass = new Vector3(0, centerOfMassHeight, 0);
            rb.mass = robotMass;
        }
    }

    void ConfigureJoint(JointConfig joint)
    {
        // Configure joint constraints based on joint type
        if (joint.jointTransform != null)
        {
            ConfigurableJoint configJoint = joint.jointTransform.GetComponent<ConfigurableJoint>();
            if (configJoint != null)
            {
                // Set joint limits based on joint type
                switch (joint.jointType)
                {
                    case JointType.Revolute:
                        configJoint.lowAngularXLimit = new SoftJointLimit { limit = joint.jointLimits.x };
                        configJoint.highAngularXLimit = new SoftJointLimit { limit = joint.jointLimits.y };
                        break;
                    case JointType.Prismatic:
                        configJoint.linearLimit = new SoftJointLimit { limit = joint.jointLimits.y };
                        break;
                }
            }
        }
    }
}

public enum JointType
{
    Revolute,
    Prismatic,
    Fixed,
    Continuous
}
```

## Physics and Collision Systems

### Configuring Physics for Humanoid Robots

```csharp
using UnityEngine;

[RequireComponent(typeof(Rigidbody))]
public class HumanoidPhysics : MonoBehaviour
{
    [Header("Balance Parameters")]
    public float balanceThreshold = 0.1f;
    public float recoveryForce = 10f;
    public float maxTorque = 100f;

    [Header("Contact Detection")]
    public LayerMask groundLayer;
    public Transform leftFoot;
    public Transform rightFoot;
    public float contactRadius = 0.1f;

    private Rigidbody rb;
    private bool leftFootContact = false;
    private bool rightFootContact = false;

    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    void FixedUpdate()
    {
        // Update contact detection
        UpdateContactDetection();

        // Apply balance forces if needed
        ApplyBalanceForces();

        // Update physics properties based on control inputs
        UpdatePhysicsControl();
    }

    void UpdateContactDetection()
    {
        // Check if feet are in contact with ground
        leftFootContact = Physics.CheckSphere(leftFoot.position, contactRadius, groundLayer);
        rightFootContact = Physics.CheckSphere(rightFoot.position, contactRadius, groundLayer);
    }

    void ApplyBalanceForces()
    {
        // Simple balance control - keep center of mass over support polygon
        Vector3 com = rb.worldCenterOfMass;
        Vector3 supportCenter = CalculateSupportCenter();

        Vector3 balanceError = com - supportCenter;
        balanceError.y = 0; // Only consider horizontal error

        if (balanceError.magnitude > balanceThreshold)
        {
            // Apply corrective force
            Vector3 correctiveForce = -balanceError.normalized * recoveryForce;
            correctiveForce = Vector3.ClampMagnitude(correctiveForce, maxTorque);

            rb.AddForceAtPosition(correctiveForce, supportCenter, ForceMode.Force);
        }
    }

    Vector3 CalculateSupportCenter()
    {
        if (leftFootContact && rightFootContact)
        {
            // Both feet on ground - use midpoint
            return (leftFoot.position + rightFoot.position) * 0.5f;
        }
        else if (leftFootContact)
        {
            // Left foot only
            return leftFoot.position;
        }
        else if (rightFootContact)
        {
            // Right foot only
            return rightFoot.position;
        }
        else
        {
            // No contact - return current position
            return transform.position;
        }
    }

    void UpdatePhysicsControl()
    {
        // Apply control forces based on ROS commands
        // This would be connected to ROS message callbacks
    }

    void OnDrawGizmosSelected()
    {
        // Visualize contact spheres
        Gizmos.color = leftFootContact ? Color.green : Color.red;
        Gizmos.DrawWireSphere(leftFoot.position, contactRadius);

        Gizmos.color = rightFootContact ? Color.green : Color.red;
        Gizmos.DrawWireSphere(rightFoot.position, contactRadius);

        // Visualize center of mass
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(rb.worldCenterOfMass, 0.05f);

        // Visualize support polygon
        if (leftFootContact && rightFootContact)
        {
            Gizmos.color = Color.blue;
            Gizmos.DrawLine(leftFoot.position, rightFoot.position);
        }
    }
}
```

### Collision Detection and Response

```csharp
using UnityEngine;

public class CollisionHandler : MonoBehaviour
{
    [Header("Collision Response")]
    public float collisionForceThreshold = 10f;
    public float collisionRecoveryTime = 0.5f;

    private float lastCollisionTime = 0f;
    private bool isRecovering = false;

    void OnCollisionEnter(Collision collision)
    {
        // Check collision force
        float collisionForce = collision.relativeVelocity.magnitude * collision.rigidbody.mass;

        if (collisionForce > collisionForceThreshold)
        {
            HandleCollision(collision, collisionForce);
        }
    }

    void HandleCollision(Collision collision, float force)
    {
        Debug.Log($"Collision detected with {collision.gameObject.name}, force: {force}");

        // Trigger collision response
        StartCoroutine(CollisionRecovery());

        // Publish collision event to ROS
        PublishCollisionEvent(collision, force);
    }

    System.Collections.IEnumerator CollisionRecovery()
    {
        isRecovering = true;
        lastCollisionTime = Time.time;

        // Apply recovery behaviors
        ApplyRecoveryBehaviors();

        yield return new WaitForSeconds(collisionRecoveryTime);
        isRecovering = false;
    }

    void ApplyRecoveryBehaviors()
    {
        // Implement collision recovery logic
        // This could include balance recovery, step adjustment, etc.
    }

    void PublishCollisionEvent(Collision collision, float force)
    {
        // Send collision information to ROS
        // This would use ROS TCP connector
    }
}
```

## High-Fidelity Visualization Techniques

### Advanced Materials and Shading

```csharp
using UnityEngine;

public class RobotMaterials : MonoBehaviour
{
    [Header("Material Properties")]
    public Material headMaterial;
    public Material torsoMaterial;
    public Material limbMaterial;
    public Material jointMaterial;

    [Header("Visual Effects")]
    public bool enableReflections = true;
    public bool enableEmission = true;
    public float emissionIntensity = 1.0f;

    void Start()
    {
        ConfigureMaterials();
    }

    void ConfigureMaterials()
    {
        // Configure head material (skin-like)
        if (headMaterial != null)
        {
            headMaterial.SetColor("_BaseColor", new Color(0.9f, 0.8f, 0.7f));
            headMaterial.SetFloat("_Smoothness", 0.5f);
            headMaterial.SetFloat("_Metallic", 0.0f);
        }

        // Configure torso material (metallic)
        if (torsoMaterial != null)
        {
            torsoMaterial.SetColor("_BaseColor", new Color(0.7f, 0.7f, 0.8f));
            torsoMaterial.SetFloat("_Smoothness", 0.7f);
            torsoMaterial.SetFloat("_Metallic", 0.3f);
        }

        // Configure limb materials
        if (limbMaterial != null)
        {
            limbMaterial.SetColor("_BaseColor", new Color(0.6f, 0.6f, 0.7f));
            limbMaterial.SetFloat("_Smoothness", 0.6f);
            limbMaterial.SetFloat("_Metallic", 0.2f);
        }

        // Configure joint materials (more metallic)
        if (jointMaterial != null)
        {
            jointMaterial.SetColor("_BaseColor", new Color(0.8f, 0.8f, 0.8f));
            jointMaterial.SetFloat("_Smoothness", 0.8f);
            jointMaterial.SetFloat("_Metallic", 0.7f);
        }

        // Configure emission for status indicators
        if (enableEmission)
        {
            ConfigureEmission();
        }
    }

    void ConfigureEmission()
    {
        // Set up emission for status indicators
        if (headMaterial != null)
        {
            headMaterial.SetColor("_EmissionColor", Color.blue * emissionIntensity);
            headMaterial.EnableKeyword("_EMISSION");
        }
    }

    public void UpdateStatusIndicator(Color statusColor)
    {
        // Update status indicator color (e.g., eyes, chest panel)
        if (headMaterial != null)
        {
            headMaterial.SetColor("_EmissionColor", statusColor * emissionIntensity);
        }
    }
}
```

### Lighting and Environment Setup

```csharp
using UnityEngine;

public class EnvironmentSetup : MonoBehaviour
{
    [Header("Lighting Configuration")]
    public Light mainLight;
    public Color ambientLightColor = new Color(0.2f, 0.2f, 0.2f, 1);
    public float ambientIntensity = 0.5f;

    [Header("Environment")]
    public GameObject[] environmentObjects;
    public Material[] environmentMaterials;

    [Header("Reflection Probes")]
    public bool useReflectionProbes = true;
    public float reflectionUpdateInterval = 1.0f;

    void Start()
    {
        SetupLighting();
        SetupEnvironment();
        SetupReflections();
    }

    void SetupLighting()
    {
        // Configure main directional light
        if (mainLight != null)
        {
            mainLight.type = LightType.Directional;
            mainLight.color = Color.white;
            mainLight.intensity = 1.0f;
            mainLight.shadows = LightShadows.Soft;
            mainLight.shadowResolution = ShadowResolution.High;
        }

        // Set ambient lighting
        RenderSettings.ambientLight = ambientLightColor;
        RenderSettings.ambientIntensity = ambientIntensity;
    }

    void SetupEnvironment()
    {
        // Configure environment objects
        foreach (GameObject obj in environmentObjects)
        {
            if (obj != null)
            {
                // Add physics properties if needed
                Rigidbody rb = obj.GetComponent<Rigidbody>();
                if (rb == null)
                {
                    rb = obj.AddComponent<Rigidbody>();
                    rb.isKinematic = true; // Static objects
                }

                // Add collision if needed
                if (obj.GetComponent<Collider>() == null)
                {
                    obj.AddComponent<BoxCollider>();
                }
            }
        }
    }

    void SetupReflections()
    {
        if (useReflectionProbes)
        {
            // Add reflection probes for realistic reflections
            AddReflectionProbes();
        }
    }

    void AddReflectionProbes()
    {
        // Add reflection probes at strategic locations
        GameObject probeObject = new GameObject("Reflection Probe");
        ReflectionProbe probe = probeObject.AddComponent<ReflectionProbe>();

        probe.mode = ReflectionProbeMode.Realtime;
        probe.size = new Vector3(10, 10, 10);
        probe.center = Vector3.zero;
        probe.intensity = 1.0f;
        probe.refreshMode = ReflectionProbeRefreshMode.OnAwake;

        // Update reflections periodically
        InvokeRepeating("UpdateReflections", 0, reflectionUpdateInterval);
    }

    void UpdateReflections()
    {
        // Force reflection probe updates if needed
        ReflectionProbe[] probes = FindObjectsOfType<ReflectionProbe>();
        foreach (ReflectionProbe probe in probes)
        {
            probe.RenderProbe();
        }
    }
}
```

## Unity-ROS Integration

### ROS Communication Setup

```csharp
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;

public class UnityROSIntegration : MonoBehaviour
{
    [Header("ROS Connection")]
    public string rosHostname = "127.0.0.1";
    public int rosPort = 10000;

    [Header("Topics")]
    public string jointStateTopic = "joint_states";
    public string tfTopic = "tf";
    public string imuTopic = "imu/data";

    private ROSConnection ros;
    private Dictionary<string, float> jointPositions = new Dictionary<string, float>();
    private Dictionary<string, float> jointVelocities = new Dictionary<string, float>();
    private Dictionary<string, float> jointEfforts = new Dictionary<string, float>();

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();
        ros.hostname = rosHostname;
        ros.port = rosPort;

        // Subscribe to ROS topics
        ros.Subscribe<sensor_msgs.JointStateMsg>(jointStateTopic, JointStateCallback);

        // Start publishing loop
        StartCoroutine(PublishLoop());
    }

    void JointStateCallback(JointStateMsg msg)
    {
        // Update joint positions from ROS
        for (int i = 0; i < msg.name.Count; i++)
        {
            if (i < msg.position.Count)
                jointPositions[msg.name[i]] = msg.position[i];

            if (i < msg.velocity.Count)
                jointVelocities[msg.name[i]] = msg.velocity[i];

            if (i < msg.effort.Count)
                jointEfforts[msg.name[i]] = msg.effort[i];
        }
    }

    IEnumerator PublishLoop()
    {
        while (true)
        {
            // Publish current robot state
            PublishRobotState();
            PublishTF();
            PublishIMUData();

            yield return new WaitForSeconds(0.01f); // 100Hz
        }
    }

    void PublishRobotState()
    {
        var jointState = new JointStateMsg();
        jointState.header = new HeaderMsg();
        jointState.header.stamp = new TimeMsg(ROSConnection.GetTime());
        jointState.header.frame_id = "base_link";

        // Add joint names and current positions
        foreach (var joint in jointPositions)
        {
            jointState.name.Add(joint.Key);
            jointState.position.Add(joint.Value);

            // Add velocity and effort if available
            float velocity = 0f;
            float effort = 0f;

            jointVelocities.TryGetValue(joint.Key, out velocity);
            jointEfforts.TryGetValue(joint.Key, out effort);

            jointState.velocity.Add(velocity);
            jointState.effort.Add(effort);
        }

        ros.Publish(jointStateTopic, jointState);
    }

    void PublishTF()
    {
        // Publish transform messages for robot links
        // This would include all the transforms for the robot hierarchy
    }

    void PublishIMUData()
    {
        // Publish IMU data if robot has IMU sensors
    }
}
```

### Sensor Simulation in Unity

```csharp
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;

public class UnityCameraSensor : MonoBehaviour
{
    [Header("Camera Configuration")]
    public Camera sensorCamera;
    public string topicName = "camera/image_raw";
    public int width = 640;
    public int height = 480;
    public float updateRate = 30f;

    [Header("Noise Parameters")]
    public bool addNoise = true;
    public float noiseIntensity = 0.01f;

    private ROSConnection ros;
    private RenderTexture renderTexture;
    private Texture2D texture2D;
    private byte[] rawImageBytes;

    void Start()
    {
        ros = ROSConnection.GetOrCreateInstance();

        // Create render texture for camera
        renderTexture = new RenderTexture(width, height, 24);
        sensorCamera.targetTexture = renderTexture;

        // Create texture2D for reading pixels
        texture2D = new Texture2D(width, height, TextureFormat.RGB24, false);

        StartCoroutine(CaptureAndPublish());
    }

    IEnumerator CaptureAndPublish()
    {
        while (true)
        {
            yield return new WaitForEndOfFrame();
            CaptureImage();
        }
    }

    void CaptureImage()
    {
        // Copy render texture to texture2D
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        texture2D.Apply();

        // Add noise if enabled
        if (addNoise)
        {
            AddNoiseToTexture(texture2D);
        }

        // Convert to ROS image message
        byte[] imageData = texture2D.EncodeToJPG();

        // Create and publish ROS image message
        var imageMsg = new sensor_msgs.ImageMsg();
        imageMsg.header = new std_msgs.HeaderMsg();
        imageMsg.header.stamp = new builtin_interfaces.TimeMsg(ROSConnection.GetTime());
        imageMsg.header.frame_id = "camera_optical_frame";
        imageMsg.height = (uint)height;
        imageMsg.width = (uint)width;
        imageMsg.encoding = "rgb8";
        imageMsg.is_bigendian = 0;
        imageMsg.step = (uint)(width * 3); // 3 bytes per pixel for RGB
        imageMsg.data = imageData;

        ros.Publish(topicName, imageMsg);
    }

    void AddNoiseToTexture(Texture2D tex)
    {
        Color[] pixels = tex.GetPixels();

        for (int i = 0; i < pixels.Length; i++)
        {
            Color originalColor = pixels[i];

            // Add Gaussian noise
            float noiseR = Random.Range(-noiseIntensity, noiseIntensity);
            float noiseG = Random.Range(-noiseIntensity, noiseIntensity);
            float noiseB = Random.Range(-noiseIntensity, noiseIntensity);

            pixels[i] = new Color(
                Mathf.Clamp01(originalColor.r + noiseR),
                Mathf.Clamp01(originalColor.g + noiseG),
                Mathf.Clamp01(originalColor.b + noiseB)
            );
        }

        tex.SetPixels(pixels);
        tex.Apply();
    }
}
```

## Synthetic Data Generation

### Unity Perception Package

Unity Perception provides tools for generating synthetic training data:

```csharp
using UnityEngine;
using Unity.Perception.GroundTruth;
using Unity.Perception.GroundTruth.DataModel;
using Unity.Perception.Randomization.Parameters;
using Unity.Perception.Randomization.Samplers;

public class SyntheticDataGenerator : MonoBehaviour
{
    [Header("Dataset Configuration")]
    public string datasetName = "humanoid_robot_dataset";
    public int sequenceLength = 100;
    public int framesPerSecond = 10;

    [Header("Domain Randomization")]
    public bool enableDomainRandomization = true;
    public float lightingVariation = 0.5f;
    public float textureVariation = 0.3f;
    public float backgroundVariation = 0.4f;

    [Header("Annotation Settings")]
    public bool generateBoundingBoxes = true;
    public bool generateSegmentation = true;
    public bool generateDepth = true;

    private int currentFrame = 0;
    private bool isGenerating = false;

    void Start()
    {
        ConfigurePerception();
    }

    void ConfigurePerception()
    {
        // Enable perception camera
        var perceptionCamera = GetComponent<PerceptionCamera>();
        if (perceptionCamera == null)
        {
            perceptionCamera = gameObject.AddComponent<PerceptionCamera>();
        }

        // Configure annotation managers
        if (generateBoundingBoxes)
        {
            var bboxManager = GetComponent<BoundingBox2DManager>();
            if (bboxManager == null)
                gameObject.AddComponent<BoundingBox2DManager>();
        }

        if (generateSegmentation)
        {
            var segManager = GetComponent<SegmentationManager>();
            if (segManager == null)
                gameObject.AddComponent<SegmentationManager>();
        }

        if (generateDepth)
        {
            var depthManager = GetComponent<DepthManager>();
            if (depthManager == null)
                gameObject.AddComponent<DepthManager>();
        }
    }

    public void StartDataGeneration()
    {
        isGenerating = true;
        currentFrame = 0;
        StartCoroutine(GenerateDataSequence());
    }

    System.Collections.IEnumerator GenerateDataSequence()
    {
        while (currentFrame < sequenceLength && isGenerating)
        {
            // Apply domain randomization
            if (enableDomainRandomization)
            {
                ApplyDomainRandomization();
            }

            // Wait for next frame
            yield return new WaitForSeconds(1f / framesPerSecond);

            currentFrame++;
        }

        isGenerating = false;
        Debug.Log($"Data generation completed. Generated {currentFrame} frames.");
    }

    void ApplyDomainRandomization()
    {
        // Vary lighting conditions
        if (RenderSettings.sun != null)
        {
            var newIntensity = 1f + GaussianSampler.Samples(0, lightingVariation)[0];
            RenderSettings.sun.intensity = Mathf.Clamp(newIntensity, 0.5f, 2f);
        }

        // Vary background textures
        ApplyBackgroundVariation();

        // Vary robot appearance slightly
        ApplyRobotVariation();
    }

    void ApplyBackgroundVariation()
    {
        // Change background materials, colors, or objects
        // This would involve changing environment objects
    }

    void ApplyRobotVariation()
    {
        // Slightly vary robot appearance for domain randomization
        // Change colors, materials, or minor geometric variations
    }
}
```

## Performance Optimization

### Level of Detail (LOD) for Robot Models

```csharp
using UnityEngine;

public class RobotLODManager : MonoBehaviour
{
    [Header("LOD Configuration")]
    public LODGroup lodGroup;
    public Transform[] lodLevels; // Different detail levels
    public float[] screenPercentages = {0.5f, 0.2f, 0.05f}; // When to switch LODs

    [Header("Performance Settings")]
    public bool enableLOD = true;
    public bool optimizeShadows = true;
    public int maxLODLevel = 2;

    void Start()
    {
        SetupLOD();
    }

    void SetupLOD()
    {
        if (lodGroup == null)
        {
            lodGroup = gameObject.AddComponent<LODGroup>();
        }

        // Create LOD levels
        LOD[] lods = new LOD[lodLevels.Length];
        for (int i = 0; i < lodLevels.Length; i++)
        {
            Renderer[] renderers = { lodLevels[i].GetComponent<Renderer>() };
            lods[i] = new LOD(screenPercentages[i], renderers);
        }

        lodGroup.SetLODs(lods);
        lodGroup.RecalculateBounds();
    }

    void Update()
    {
        if (enableLOD && lodGroup != null)
        {
            // Adjust LOD based on performance or distance
            AdjustLODBasedOnPerformance();
        }
    }

    void AdjustLODBasedOnPerformance()
    {
        // Monitor performance and adjust LOD accordingly
        float frameTime = Time.deltaTime;
        if (frameTime > 0.033f) // Targeting 30 FPS
        {
            // Performance is low, increase LOD level (lower detail)
            lodGroup.ForceLOD(maxLODLevel);
        }
        else
        {
            // Performance is good, use lower LOD level (higher detail)
            lodGroup.ForceLOD(0);
        }
    }
}
```

### Occlusion Culling and Rendering Optimization

```csharp
using UnityEngine;

public class RenderingOptimizer : MonoBehaviour
{
    [Header("Occlusion Culling")]
    public bool enableOcclusionCulling = true;
    public bool enableFrustumCulling = true;

    [Header("Batching")]
    public bool enableDynamicBatching = true;
    public bool enableStaticBatching = true;

    [Header("Shader Optimization")]
    public bool useMobileShaders = false;
    public bool enableLODForShaders = true;

    void Start()
    {
        ConfigureRenderingOptimization();
    }

    void ConfigureRenderingOptimization()
    {
        // Configure occlusion culling
        if (enableOcclusionCulling)
        {
            // This is typically done in Unity Editor through Static flags
            // Mark environment objects as static for occlusion culling
        }

        // Configure batching
        QualitySettings.pixelLightCount = 2; // Limit dynamic lights
        QualitySettings.anisotropicFiltering = AnisotropicFiltering.Enable;

        // Configure shader variants
        ConfigureShaders();
    }

    void ConfigureShaders()
    {
        if (useMobileShaders)
        {
            // Use simplified shaders for better performance
            Shader.globalMaximumLOD = 300; // Use simpler shader variants
        }
        else
        {
            Shader.globalMaximumLOD = 600; // Use higher quality shaders
        }
    }

    void OnBecameVisible()
    {
        // Called when object becomes visible
        // Can enable more detailed rendering
    }

    void OnBecameInvisible()
    {
        // Called when object becomes invisible
        // Can disable expensive rendering features
    }
}
```

## Best Practices for Unity Robotics

### 1. Physics Accuracy vs. Performance
- Use appropriate physics settings for your use case
- Balance visual quality with computational requirements
- Test performance on target hardware

### 2. Coordinate System Consistency
- Ensure Unity coordinate system matches ROS conventions
- Use proper TF transforms between Unity and ROS frames
- Maintain consistent units (meters, radians, etc.)

### 3. Real-time Communication
- Optimize ROS message frequency for real-time performance
- Use efficient serialization for large data (images, point clouds)
- Implement proper error handling for network interruptions

### 4. Asset Management
- Use efficient 3D models optimized for real-time rendering
- Implement proper texture compression
- Use object pooling for frequently instantiated objects

## Summary

This chapter covered Unity integration for high-fidelity robot visualization, including setup, physics configuration, sensor simulation, and performance optimization. Unity provides powerful visualization capabilities that complement traditional robotics simulators, particularly for perception training and human-robot interaction studies. In the next chapter, we'll explore environment and scenario building.

## Exercises

1. Set up a basic Unity scene with a humanoid robot model
2. Implement ROS communication for joint state control
3. Create a realistic environment with proper lighting
4. Implement synthetic data generation for computer vision training
5. Optimize the Unity scene for real-time performance

## Further Reading

- Unity Robotics Hub Documentation: https://github.com/Unity-Technologies/Unity-Robotics-Hub
- Unity Perception Package: https://docs.unity3d.com/Packages/com.unity.perception@latest
- NVIDIA Isaac Unity Robotics: https://developer.nvidia.com/isaac-unity-robotics

---

*Next: [Chapter 5: Environment & Scenario Building](../chapter5-environment-building/README.md)*