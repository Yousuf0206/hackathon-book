# Chapter 5: Environment & Scenario Building

## Learning Objectives

After completing this chapter, you will be able to:
- Design and create diverse simulation environments for humanoid robotics
- Build complex scenarios for testing robot capabilities
- Implement dynamic environments with interactive elements
- Create reproducible test scenarios for validation
- Design environments that bridge the sim-to-real gap

## Introduction to Environment Design for Humanoid Robots

Environment design is a critical aspect of digital twin technology for humanoid robotics. The environments in which humanoid robots operate significantly impact their perception, navigation, manipulation, and social interaction capabilities. Well-designed environments enable comprehensive testing of robot systems while providing realistic challenges that mirror real-world conditions.

### Key Considerations for Humanoid Environments

Humanoid robots operate in human-centric environments, which present unique challenges:

- **Scale**: Environments designed for human proportions and capabilities
- **Furniture**: Tables, chairs, doors, and other human-scale objects
- **Navigation**: Doorways, stairs, and pathways sized for humans
- **Interaction**: Objects designed for human manipulation
- **Social Context**: Environments that support human-robot interaction

## Environment Categories for Humanoid Robotics

### Indoor Environments

#### Home Environments
Home environments test basic domestic capabilities:

```csharp
// Unity environment configuration for home setting
using UnityEngine;

[CreateAssetMenu(fileName = "HomeEnvironment", menuName = "Robotics/Environment/Home")]
public class HomeEnvironmentConfig : ScriptableObject
{
    [Header("Room Layout")]
    public Vector2Int roomSize = new Vector2Int(8, 6); // in meters
    public bool hasKitchen = true;
    public bool hasLivingRoom = true;
    public bool hasBedroom = true;
    public bool hasBathroom = true;

    [Header("Furniture Configuration")]
    public GameObject[] furniturePrefabs;
    public float minFurnitureSize = 0.3f;
    public float maxFurnitureSize = 2.0f;

    [Header("Navigation Constraints")]
    public float doorWidth = 0.8f;
    public float tableHeight = 0.75f;
    public float chairHeight = 0.45f;

    [Header("Interaction Objects")]
    public GameObject[] kitchenObjects; // cups, plates, utensils
    public GameObject[] livingRoomObjects; // books, remotes, decorations
    public GameObject[] bedroomObjects; // clothes, bed items
}
```

#### Office Environments
Office environments test professional interaction capabilities:

```csharp
[CreateAssetMenu(fileName = "OfficeEnvironment", menuName = "Robotics/Environment/Office")]
public class OfficeEnvironmentConfig : ScriptableObject
{
    [Header("Office Layout")]
    public int numDesks = 6;
    public int numMeetingRooms = 2;
    public bool hasReception = true;
    public bool hasKitchenette = true;

    [Header("Office Equipment")]
    public GameObject[] officeEquipment; // computers, printers, phones
    public GameObject[] furniture; // desks, chairs, filing cabinets

    [Header("Navigation Features")]
    public float hallwayWidth = 1.2f;
    public float deskSpacing = 2.0f;
    public bool hasElevator = false;
    public bool hasStairs = false;
}
```

### Outdoor Environments

#### Urban Environments
Urban environments test navigation and interaction in city settings:

```csharp
[CreateAssetMenu(fileName = "UrbanEnvironment", menuName = "Robotics/Environment/Urban")]
public class UrbanEnvironmentConfig : ScriptableObject
{
    [Header("Street Layout")]
    public int numLanes = 2;
    public float laneWidth = 3.5f;
    public float sidewalkWidth = 2.0f;
    public bool hasCrosswalks = true;
    public bool hasTrafficLights = true;

    [Header("Urban Features")]
    public GameObject[] streetFurniture; // benches, trash cans, street lights
    public GameObject[] buildings; // shops, offices, residential
    public bool hasParks = false;
    public bool hasPublicTransport = false;

    [Header("Navigation Challenges")]
    public float maxSlope = 0.1f; // maximum ground slope
    public bool hasCurbRamps = true;
    public bool hasPedestrianCrossings = true;
}
```

## Environment Building Techniques

### Procedural Environment Generation

Procedural generation allows for creating diverse environments automatically:

```csharp
using System.Collections.Generic;
using UnityEngine;

public class ProceduralEnvironmentBuilder : MonoBehaviour
{
    [Header("Generation Parameters")]
    public EnvironmentConfig environmentConfig;
    public int seed = 12345;
    public bool randomizeLayout = true;

    [Header("Prefab Libraries")]
    public GameObject[] wallPrefabs;
    public GameObject[] floorPrefabs;
    public GameObject[] furniturePrefabs;
    public GameObject[] obstaclePrefabs;

    private System.Random random;

    void Start()
    {
        random = new System.Random(seed);
        BuildEnvironment();
    }

    void BuildEnvironment()
    {
        // Generate basic structure
        GenerateStructure();

        // Add furniture and objects
        AddFurniture();

        // Add obstacles and challenges
        AddObstacles();

        // Configure navigation mesh
        ConfigureNavigation();
    }

    void GenerateStructure()
    {
        // Create rooms based on configuration
        if (environmentConfig.hasKitchen)
            CreateRoom("Kitchen", new Vector2(4, 3), new Vector3(-3, 0, 0));

        if (environmentConfig.hasLivingRoom)
            CreateRoom("LivingRoom", new Vector2(5, 4), new Vector3(0, 0, 0));

        if (environmentConfig.hasBedroom)
            CreateRoom("Bedroom", new Vector2(4, 3), new Vector3(4, 0, 0));

        // Add connecting hallways
        CreateHallway(new Vector3(-1, 0, 0), new Vector2(2, 1));
    }

    GameObject CreateRoom(string name, Vector2 size, Vector3 position)
    {
        // Create room object
        GameObject room = new GameObject(name);
        room.transform.position = position;

        // Add floor
        GameObject floor = Instantiate(
            floorPrefabs[random.Next(floorPrefabs.Length)],
            position + new Vector3(size.x/2, -0.01f, size.y/2),
            Quaternion.identity);
        floor.transform.localScale = new Vector3(size.x, 1, size.y);
        floor.transform.SetParent(room.transform);

        // Add walls
        CreateWalls(room, size, position);

        return room;
    }

    void CreateWalls(GameObject parent, Vector2 size, Vector3 position)
    {
        // Create four walls for the room
        float wallHeight = 2.5f;

        // Left wall
        GameObject leftWall = Instantiate(
            wallPrefabs[random.Next(wallPrefabs.Length)],
            position + new Vector3(0, wallHeight/2, size.y/2),
            Quaternion.Euler(0, 90, 0));
        leftWall.transform.localScale = new Vector3(wallHeight, 1, size.y);
        leftWall.transform.SetParent(parent.transform);

        // Right wall
        GameObject rightWall = Instantiate(
            wallPrefabs[random.Next(wallPrefabs.Length)],
            position + new Vector3(size.x, wallHeight/2, size.y/2),
            Quaternion.Euler(0, 90, 0));
        rightWall.transform.localScale = new Vector3(wallHeight, 1, size.y);
        rightWall.transform.SetParent(parent.transform);

        // Front wall
        GameObject frontWall = Instantiate(
            wallPrefabs[random.Next(wallPrefabs.Length)],
            position + new Vector3(size.x/2, wallHeight/2, 0),
            Quaternion.identity);
        frontWall.transform.localScale = new Vector3(size.x, 1, wallHeight);
        frontWall.transform.SetParent(parent.transform);

        // Back wall
        GameObject backWall = Instantiate(
            wallPrefabs[random.Next(wallPrefabs.Length)],
            position + new Vector3(size.x/2, wallHeight/2, size.y),
            Quaternion.identity);
        backWall.transform.localScale = new Vector3(size.x, 1, wallHeight);
        backWall.transform.SetParent(parent.transform);
    }

    void AddFurniture()
    {
        // Place furniture based on room type
        foreach (Transform room in transform)
        {
            string roomType = room.name;
            int furnitureCount = random.Next(2, 6);

            for (int i = 0; i < furnitureCount; i++)
            {
                Vector3 spawnPos = GetRandomPositionInRoom(room);
                GameObject furniture = Instantiate(
                    furniturePrefabs[random.Next(furniturePrefabs.Length)],
                    spawnPos,
                    Quaternion.identity);
                furniture.transform.SetParent(room);
            }
        }
    }

    Vector3 GetRandomPositionInRoom(Transform room)
    {
        // Get room bounds and return random position within them
        // This is a simplified version - in practice, you'd check for collisions
        Vector3 roomPos = room.position;
        Vector3 size = new Vector3(4, 0, 3); // Approximate room size
        float x = roomPos.x + (float)random.NextDouble() * size.x;
        float z = roomPos.z + (float)random.NextDouble() * size.z;
        return new Vector3(x, 0.1f, z); // Slightly above ground
    }

    void AddObstacles()
    {
        // Add dynamic obstacles for testing navigation
        int obstacleCount = random.Next(3, 8);
        for (int i = 0; i < obstacleCount; i++)
        {
            Vector3 pos = new Vector3(
                (float)random.NextDouble() * 10 - 5, // Random position in environment
                0.5f,
                (float)random.NextDouble() * 8 - 4);

            GameObject obstacle = Instantiate(
                obstaclePrefabs[random.Next(obstaclePrefabs.Length)],
                pos,
                Quaternion.identity);
        }
    }

    void ConfigureNavigation()
    {
        // Bake navigation mesh for pathfinding
        // This would typically be done in the Unity Editor
        // or using runtime navmesh building
    }
}
```

### Modular Environment Building

Modular building allows for reusable environment components:

```csharp
using UnityEngine;
using System.Collections.Generic;

public class ModularEnvironment : MonoBehaviour
{
    [Header("Modular Components")]
    public RoomModule[] roomModules;
    public CorridorModule[] corridorModules;
    public FurnitureModule[] furnitureModules;

    [System.Serializable]
    public class RoomModule
    {
        public string moduleName;
        public GameObject modulePrefab;
        public Vector2Int size;
        public List<ConnectionPoint> connectionPoints;
    }

    [System.Serializable]
    public class CorridorModule
    {
        public string moduleName;
        public GameObject modulePrefab;
        public float length;
    }

    [System.Serializable]
    public class FurnitureModule
    {
        public string furnitureName;
        public GameObject furniturePrefab;
        public FurnitureType furnitureType;
    }

    public enum FurnitureType
    {
        Seating,
        Storage,
        WorkSurface,
        Decoration,
        Appliance
    }

    [System.Serializable]
    public class ConnectionPoint
    {
        public string name;
        public Vector3 position;
        public Vector3 direction;
        public ConnectionType type;
    }

    public enum ConnectionType
    {
        Doorway,
        Hallway,
        Stairs,
        Elevator
    }

    public void BuildEnvironmentFromModules(List<RoomModule> selectedRooms)
    {
        // Clear existing environment
        foreach (Transform child in transform)
        {
            DestroyImmediate(child.gameObject);
        }

        // Place selected rooms
        Vector3 currentPos = Vector3.zero;
        float offsetX = 0;

        foreach (RoomModule room in selectedRooms)
        {
            GameObject roomInstance = Instantiate(room.modulePrefab, currentPos, Quaternion.identity);
            roomInstance.transform.SetParent(transform);

            // Position next room
            offsetX += room.size.x + 1; // Add 1m gap between rooms
            currentPos = new Vector3(offsetX, 0, 0);
        }

        // Connect rooms with corridors
        ConnectRooms(selectedRooms);
    }

    void ConnectRooms(List<RoomModule> rooms)
    {
        // Implementation to connect rooms with corridors
        // This would involve finding connection points and placing corridor modules
    }
}
```

## Dynamic and Interactive Environments

### Physics-Based Interactions

Creating environments where robots can interact with objects:

```csharp
using UnityEngine;

public class InteractiveEnvironment : MonoBehaviour
{
    [Header("Interactive Objects")]
    public InteractiveObject[] interactiveObjects;
    public float interactionDistance = 1.0f;

    [Header("Physics Properties")]
    public float gravityScale = 1.0f;
    public float frictionCoefficient = 0.5f;

    [Header("Dynamic Elements")]
    public bool hasMovingObstacles = true;
    public float obstacleSpeed = 0.5f;

    [System.Serializable]
    public class InteractiveObject
    {
        public string objectName;
        public GameObject objectPrefab;
        public InteractionType interactionType;
        public bool isMovable;
        public bool isGraspable;
        public bool isStackable;
    }

    public enum InteractionType
    {
        Pushable,
        Graspable,
        Stackable,
        Openable, // doors, drawers
        Operable  // switches, buttons
    }

    void Start()
    {
        SetupInteractiveEnvironment();
    }

    void SetupInteractiveEnvironment()
    {
        // Configure physics for all interactive objects
        foreach (InteractiveObject obj in interactiveObjects)
        {
            GameObject instance = Instantiate(obj.objectPrefab);

            // Add physics components based on interaction type
            Rigidbody rb = instance.AddComponent<Rigidbody>();
            rb.useGravity = true;
            rb.drag = 1.0f;
            rb.angularDrag = 1.0f;

            // Configure collision properties
            Collider col = instance.GetComponent<Collider>();
            if (col != null)
            {
                col.material = new PhysicMaterial();
                col.material.staticFriction = frictionCoefficient;
                col.material.dynamicFriction = frictionCoefficient;
                col.material.bounciness = 0.1f;
            }

            // Add interaction script
            ObjectInteraction interaction = instance.AddComponent<ObjectInteraction>();
            interaction.Configure(obj);
        }

        // Setup moving obstacles if enabled
        if (hasMovingObstacles)
        {
            SetupMovingObstacles();
        }
    }

    void SetupMovingObstacles()
    {
        // Create obstacles that move along predefined paths
        int obstacleCount = 5;
        for (int i = 0; i < obstacleCount; i++)
        {
            GameObject obstacle = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            obstacle.name = $"MovingObstacle_{i}";

            // Set initial position
            obstacle.transform.position = new Vector3(
                Random.Range(-5f, 5f),
                0.5f,
                Random.Range(-3f, 3f));

            // Add movement script
            MovingObstacle mover = obstacle.AddComponent<MovingObstacle>();
            mover.speed = obstacleSpeed;
            mover.path = GenerateRandomPath(obstacle.transform.position);
        }
    }

    List<Vector3> GenerateRandomPath(Vector3 startPos)
    {
        List<Vector3> path = new List<Vector3>();
        path.Add(startPos);

        // Generate a simple path with 3-5 waypoints
        int waypoints = Random.Range(3, 6);
        Vector3 currentPos = startPos;

        for (int i = 0; i < waypoints; i++)
        {
            Vector3 nextPos = currentPos + new Vector3(
                Random.Range(-2f, 2f),
                0,
                Random.Range(-2f, 2f));
            path.Add(nextPos);
            currentPos = nextPos;
        }

        return path;
    }
}

public class ObjectInteraction : MonoBehaviour
{
    private InteractiveEnvironment.InteractiveObject config;

    public void Configure(InteractiveEnvironment.InteractiveObject objConfig)
    {
        config = objConfig;
    }

    void OnTriggerEnter(Collider other)
    {
        // Check if the collider is from a robot
        if (other.CompareTag("Robot"))
        {
            HandleRobotInteraction(other.gameObject);
        }
    }

    void HandleRobotInteraction(GameObject robot)
    {
        switch (config.interactionType)
        {
            case InteractiveEnvironment.InteractionType.Pushable:
                ApplyPushForce(robot);
                break;
            case InteractiveEnvironment.InteractionType.Graspable:
                PrepareForGrasping(robot);
                break;
            case InteractiveEnvironment.InteractionType.Openable:
                HandleOpenable(robot);
                break;
        }
    }

    void ApplyPushForce(GameObject robot)
    {
        // Apply force based on robot's push action
        Vector3 pushDirection = (transform.position - robot.transform.position).normalized;
        Rigidbody rb = GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.AddForce(pushDirection * 10f, ForceMode.Impulse);
        }
    }

    void PrepareForGrasping(GameObject robot)
    {
        // Prepare object for grasping (disable collisions temporarily, etc.)
        // This would interface with robot's grasping system
    }

    void HandleOpenable(GameObject robot)
    {
        // Handle opening actions (doors, drawers, etc.)
        // This might involve animation or state changes
    }
}

public class MovingObstacle : MonoBehaviour
{
    public List<Vector3> path;
    public float speed = 1.0f;
    private int currentTargetIndex = 0;
    private bool movingForward = true;

    void Update()
    {
        if (path.Count == 0) return;

        // Move towards current target
        Vector3 target = path[currentTargetIndex];
        transform.position = Vector3.MoveTowards(
            transform.position,
            target,
            speed * Time.deltaTime);

        // Check if reached target
        if (Vector3.Distance(transform.position, target) < 0.1f)
        {
            UpdateTarget();
        }
    }

    void UpdateTarget()
    {
        if (path.Count <= 1) return;

        if (movingForward)
        {
            currentTargetIndex++;
            if (currentTargetIndex >= path.Count - 1)
            {
                movingForward = false;
            }
        }
        else
        {
            currentTargetIndex--;
            if (currentTargetIndex <= 0)
            {
                movingForward = true;
            }
        }
    }
}
```

### Environmental Dynamics

Creating environments with changing conditions:

```csharp
using UnityEngine;
using System.Collections;

public class EnvironmentalDynamics : MonoBehaviour
{
    [Header("Lighting Dynamics")]
    public bool simulateDayNight = true;
    public float dayNightCycleSpeed = 0.1f;
    public AnimationCurve dayNightIntensity;

    [Header("Weather Simulation")]
    public bool enableWeather = false;
    public WeatherType currentWeather = WeatherType.Clear;
    public float weatherChangeInterval = 30f; // seconds

    [Header("Crowd Simulation")]
    public bool simulateCrowds = false;
    public int minCrowdSize = 5;
    public int maxCrowdSize = 15;
    public float crowdMovementSpeed = 1.0f;

    [Header("Event Triggers")]
    public bool enableRandomEvents = true;
    public float eventInterval = 60f; // seconds

    public enum WeatherType
    {
        Clear,
        Cloudy,
        Rainy,
        Foggy
    }

    private Light sunLight;
    private float dayNightTimer = 0f;
    private float weatherTimer = 0f;
    private float eventTimer = 0f;

    void Start()
    {
        sunLight = GameObject.FindGameObjectWithTag("Sun")?.GetComponent<Light>();
        StartCoroutine(EnvironmentalUpdateLoop());
    }

    IEnumerator EnvironmentalUpdateLoop()
    {
        while (true)
        {
            UpdateDayNightCycle();
            UpdateWeather();
            UpdateCrowdSimulation();
            CheckForRandomEvents();

            yield return new WaitForSeconds(1f); // Update every second
        }
    }

    void UpdateDayNightCycle()
    {
        if (simulateDayNight && sunLight != null)
        {
            dayNightTimer += dayNightCycleSpeed * Time.deltaTime;

            // Normalize to 0-1 range (24 hours)
            float dayProgress = (dayNightTimer % 24f) / 24f;

            // Apply day/night intensity curve
            float intensity = dayNightIntensity.Evaluate(dayProgress);
            sunLight.intensity = Mathf.Lerp(0.1f, 1.0f, intensity);

            // Rotate sun to simulate time of day
            sunLight.transform.rotation = Quaternion.Euler(
                90 - (dayProgress * 360), // Elevation
                dayProgress * 360 - 90,   // Azimuth
                0
            );

            // Change color based on time of day
            if (dayProgress < 0.25f || dayProgress > 0.75f)
            {
                // Night - cooler colors
                sunLight.color = Color.Lerp(new Color(0.2f, 0.2f, 0.4f), Color.white, intensity);
            }
            else if (dayProgress > 0.25f && dayProgress < 0.3f)
            {
                // Sunrise - warm colors
                sunLight.color = Color.Lerp(new Color(1.0f, 0.7f, 0.3f), Color.white,
                    Mathf.InverseLerp(0.25f, 0.3f, dayProgress));
            }
            else if (dayProgress > 0.7f && dayProgress < 0.75f)
            {
                // Sunset - warm colors
                sunLight.color = Color.Lerp(new Color(1.0f, 0.5f, 0.2f), Color.white,
                    Mathf.InverseLerp(0.75f, 0.7f, dayProgress));
            }
        }
    }

    void UpdateWeather()
    {
        if (enableWeather)
        {
            weatherTimer += Time.deltaTime;
            if (weatherTimer >= weatherChangeInterval)
            {
                ChangeWeather();
                weatherTimer = 0f;
            }
        }
    }

    void ChangeWeather()
    {
        // Randomly select new weather
        WeatherType[] weatherTypes = (WeatherType[])System.Enum.GetValues(typeof(WeatherType));
        currentWeather = weatherTypes[Random.Range(0, weatherTypes.Length)];

        ApplyWeatherEffect(currentWeather);
    }

    void ApplyWeatherEffect(WeatherType weather)
    {
        switch (weather)
        {
            case WeatherType.Clear:
                RenderSettings.fog = false;
                if (sunLight != null) sunLight.intensity = 1.0f;
                break;
            case WeatherType.Cloudy:
                RenderSettings.fog = true;
                RenderSettings.fogColor = new Color(0.7f, 0.7f, 0.8f);
                RenderSettings.fogDensity = 0.01f;
                if (sunLight != null) sunLight.intensity = 0.7f;
                break;
            case WeatherType.Rainy:
                RenderSettings.fog = true;
                RenderSettings.fogColor = new Color(0.5f, 0.5f, 0.6f);
                RenderSettings.fogDensity = 0.05f;
                if (sunLight != null) sunLight.intensity = 0.5f;
                CreateRainEffect();
                break;
            case WeatherType.Foggy:
                RenderSettings.fog = true;
                RenderSettings.fogColor = Color.gray;
                RenderSettings.fogDensity = 0.08f;
                if (sunLight != null) sunLight.intensity = 0.3f;
                break;
        }
    }

    void CreateRainEffect()
    {
        // Create rain particles
        GameObject rainSystem = new GameObject("RainSystem");
        ParticleSystem rainPS = rainSystem.AddComponent<ParticleSystem>();

        var main = rainPS.main;
        main.startLifetime = 2f;
        main.startSpeed = 10f;
        main.startSize = 0.1f;
        main.maxParticles = 1000;

        var emission = rainPS.emission;
        emission.rateOverTime = 500;

        var shape = rainPS.shape;
        shape.shapeType = ParticleSystemShapeType.Box;
        shape.scale = new Vector3(20, 1, 20); // Cover large area
    }

    void UpdateCrowdSimulation()
    {
        if (simulateCrowds)
        {
            // Spawn or update crowd NPCs
            // This would involve spawning and managing multiple character controllers
        }
    }

    void CheckForRandomEvents()
    {
        if (enableRandomEvents)
        {
            eventTimer += Time.deltaTime;
            if (eventTimer >= eventInterval)
            {
                TriggerRandomEvent();
                eventTimer = 0f;
            }
        }
    }

    void TriggerRandomEvent()
    {
        // Randomly select an event to trigger
        int eventType = Random.Range(0, 3);

        switch (eventType)
        {
            case 0:
                // Door opens/closes
                StartCoroutine(RandomDoorEvent());
                break;
            case 1:
                // Light turns on/off
                StartCoroutine(RandomLightEvent());
                break;
            case 2:
                // Object moves
                StartCoroutine(RandomObjectEvent());
                break;
        }
    }

    IEnumerator RandomDoorEvent()
    {
        // Find and animate a random door
        yield return new WaitForSeconds(0.5f);
    }

    IEnumerator RandomLightEvent()
    {
        // Find and toggle a random light
        yield return new WaitForSeconds(0.5f);
    }

    IEnumerator RandomObjectEvent()
    {
        // Move a random object
        yield return new WaitForSeconds(0.5f);
    }
}
```

## Scenario Design for Humanoid Robotics

### Testing Scenarios

Different scenarios test different capabilities:

```csharp
using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class ScenarioManager : MonoBehaviour
{
    [Header("Scenario Configuration")]
    public List<ScenarioConfig> availableScenarios;
    public ScenarioConfig currentScenario;

    [Header("Scenario Execution")]
    public bool autoExecute = false;
    public float scenarioTimeout = 300f; // 5 minutes

    [Header("Success Criteria")]
    public List<SuccessCondition> successConditions;
    public List<FailureCondition> failureConditions;

    [System.Serializable]
    public class ScenarioConfig
    {
        public string scenarioName;
        public ScenarioType scenarioType;
        public DifficultyLevel difficulty;
        public List<ScenarioTask> tasks;
        public EnvironmentConfig environment;
        public RobotConfig robotConfig;
    }

    public enum ScenarioType
    {
        Navigation,
        Manipulation,
        SocialInteraction,
        Balance,
        MultiTask
    }

    public enum DifficultyLevel
    {
        Beginner,
        Intermediate,
        Advanced,
        Expert
    }

    [System.Serializable]
    public class ScenarioTask
    {
        public string taskName;
        public TaskType taskType;
        public Vector3 targetLocation;
        public GameObject targetObject;
        public float timeLimit = 60f;
        public bool isOptional = false;
    }

    public enum TaskType
    {
        NavigateTo,
        PickUpObject,
        PlaceObject,
        OpenDoor,
        FollowPerson,
        AvoidObstacle,
        MaintainBalance
    }

    [System.Serializable]
    public class SuccessCondition
    {
        public ConditionType type;
        public float threshold;
        public string description;
    }

    public enum ConditionType
    {
        TaskCompletion,
        TimeEfficiency,
        EnergyEfficiency,
        Safety,
        Accuracy
    }

    [System.Serializable]
    public class FailureCondition
    {
        public FailureType type;
        public float threshold;
        public string description;
    }

    public enum FailureType
    {
        TimeLimitExceeded,
        RobotFall,
        ObjectDamage,
        NavigationFailure,
        SafetyViolation
    }

    private bool scenarioRunning = false;
    private float scenarioTimer = 0f;
    private List<ScenarioTask> completedTasks = new List<ScenarioTask>();

    void Start()
    {
        if (autoExecute && availableScenarios.Count > 0)
        {
            StartScenario(availableScenarios[0]);
        }
    }

    public void StartScenario(ScenarioConfig scenario)
    {
        if (scenarioRunning) return;

        currentScenario = scenario;
        scenarioRunning = true;
        scenarioTimer = 0f;
        completedTasks.Clear();

        // Setup environment for scenario
        SetupScenarioEnvironment();

        // Setup robot for scenario
        SetupRobotForScenario();

        // Start scenario execution
        StartCoroutine(ExecuteScenario());
    }

    void SetupScenarioEnvironment()
    {
        // Load or configure environment based on scenario requirements
        // This might involve spawning objects, configuring physics, etc.
    }

    void SetupRobotForScenario()
    {
        // Configure robot settings for this scenario
        // Set initial position, sensor configurations, etc.
    }

    IEnumerator ExecuteScenario()
    {
        while (scenarioRunning && scenarioTimer < scenarioTimeout)
        {
            scenarioTimer += Time.deltaTime;

            // Check success conditions
            if (CheckSuccessConditions())
            {
                EndScenario(true);
                yield break;
            }

            // Check failure conditions
            if (CheckFailureConditions())
            {
                EndScenario(false);
                yield break;
            }

            // Update scenario-specific tasks
            UpdateScenarioTasks();

            yield return new WaitForEndOfFrame();
        }

        // Timeout reached
        EndScenario(false);
    }

    bool CheckSuccessConditions()
    {
        foreach (SuccessCondition condition in successConditions)
        {
            switch (condition.type)
            {
                case ConditionType.TaskCompletion:
                    if (GetCompletedTaskCount() >= condition.threshold)
                        return true;
                    break;
                case ConditionType.TimeEfficiency:
                    if (scenarioTimer <= condition.threshold)
                        return true;
                    break;
                // Add other condition checks
            }
        }
        return false;
    }

    bool CheckFailureConditions()
    {
        foreach (FailureCondition condition in failureConditions)
        {
            switch (condition.type)
            {
                case FailureType.TimeLimitExceeded:
                    if (scenarioTimer >= scenarioTimeout)
                        return true;
                    break;
                case FailureType.RobotFall:
                    if (IsRobotFallen())
                        return true;
                    break;
                // Add other failure checks
            }
        }
        return false;
    }

    int GetCompletedTaskCount()
    {
        return completedTasks.Count;
    }

    bool IsRobotFallen()
    {
        // Check if robot has fallen (based on orientation, position, etc.)
        // This would interface with robot state monitoring
        return false;
    }

    void UpdateScenarioTasks()
    {
        // Update all active tasks in the scenario
        foreach (ScenarioTask task in currentScenario.tasks)
        {
            if (!completedTasks.Contains(task))
            {
                CheckTaskCompletion(task);
            }
        }
    }

    void CheckTaskCompletion(ScenarioTask task)
    {
        switch (task.taskType)
        {
            case TaskType.NavigateTo:
                if (IsAtLocation(task.targetLocation))
                {
                    completedTasks.Add(task);
                    Debug.Log($"Task completed: {task.taskName}");
                }
                break;
            case TaskType.PickUpObject:
                if (HasPickedUpObject(task.targetObject))
                {
                    completedTasks.Add(task);
                    Debug.Log($"Task completed: {task.taskName}");
                }
                break;
            // Add other task completion checks
        }
    }

    bool IsAtLocation(Vector3 targetLocation)
    {
        // Check if robot is at the target location
        // This would interface with robot position tracking
        return false;
    }

    bool HasPickedUpObject(GameObject targetObject)
    {
        // Check if robot has picked up the target object
        // This would interface with robot gripper/holding system
        return false;
    }

    void EndScenario(bool success)
    {
        scenarioRunning = false;

        if (success)
        {
            Debug.Log($"Scenario completed successfully! Time: {scenarioTimer:F2}s");
        }
        else
        {
            Debug.Log($"Scenario failed. Time: {scenarioTimer:F2}s");
        }

        // Cleanup scenario
        CleanupScenario();
    }

    void CleanupScenario()
    {
        // Reset environment, robot, and scenario state
    }

    public void StopScenario()
    {
        scenarioRunning = false;
        CleanupScenario();
    }
}
```

### Reproducible Test Environments

Creating environments that can be consistently reproduced:

```csharp
using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;

[System.Serializable]
public class EnvironmentSnapshot
{
    public string environmentName;
    public string timestamp;
    public string scenarioDescription;
    public List<ObjectState> objectStates;
    public List<LightState> lightStates;
    public List<PhysicsState> physicsStates;
    public EnvironmentConfig config;

    [System.Serializable]
    public class ObjectState
    {
        public string name;
        public Vector3 position;
        public Quaternion rotation;
        public Vector3 scale;
        public bool isActive;
        public string prefabName;
    }

    [System.Serializable]
    public class LightState
    {
        public string name;
        public Vector3 position;
        public Quaternion rotation;
        public Color color;
        public float intensity;
        public LightType type;
    }

    [System.Serializable]
    public class PhysicsState
    {
        public float gravityScale;
        public float timeScale;
        public bool useGravity;
    }
}

public class EnvironmentSnapshotManager : MonoBehaviour
{
    [Header("Snapshot Configuration")]
    public string snapshotDirectory = "EnvironmentSnapshots/";

    public void CreateSnapshot(string name, string description)
    {
        EnvironmentSnapshot snapshot = new EnvironmentSnapshot();
        snapshot.environmentName = name;
        snapshot.timestamp = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        snapshot.scenarioDescription = description;
        snapshot.config = GetEnvironmentConfig();

        // Capture all object states
        snapshot.objectStates = CaptureObjectStates();

        // Capture lighting states
        snapshot.lightStates = CaptureLightStates();

        // Capture physics states
        snapshot.physicsStates = CapturePhysicsStates();

        // Save snapshot
        SaveSnapshot(snapshot);
    }

    EnvironmentConfig GetEnvironmentConfig()
    {
        // Get current environment configuration
        return null; // Implementation depends on your environment system
    }

    List<EnvironmentSnapshot.ObjectState> CaptureObjectStates()
    {
        List<EnvironmentSnapshot.ObjectState> states = new List<EnvironmentSnapshot.ObjectState>();

        // Get all objects in the scene that should be captured
        GameObject[] allObjects = FindObjectsOfType<GameObject>();

        foreach (GameObject obj in allObjects)
        {
            if (ShouldIncludeInSnapshot(obj))
            {
                EnvironmentSnapshot.ObjectState state = new EnvironmentSnapshot.ObjectState();
                state.name = obj.name;
                state.position = obj.transform.position;
                state.rotation = obj.transform.rotation;
                state.scale = obj.transform.localScale;
                state.isActive = obj.activeSelf;
                state.prefabName = GetPrefabName(obj);

                states.Add(state);
            }
        }

        return states;
    }

    List<EnvironmentSnapshot.LightState> CaptureLightStates()
    {
        List<EnvironmentSnapshot.LightState> states = new List<EnvironmentSnapshot.LightState>();
        Light[] lights = FindObjectsOfType<Light>();

        foreach (Light light in lights)
        {
            EnvironmentSnapshot.LightState state = new EnvironmentSnapshot.LightState();
            state.name = light.name;
            state.position = light.transform.position;
            state.rotation = light.transform.rotation;
            state.color = light.color;
            state.intensity = light.intensity;
            state.type = light.type;

            states.Add(state);
        }

        return states;
    }

    List<EnvironmentSnapshot.PhysicsState> CapturePhysicsStates()
    {
        List<EnvironmentSnapshot.PhysicsState> states = new List<EnvironmentSnapshot.PhysicsState>();

        EnvironmentSnapshot.PhysicsState state = new EnvironmentSnapshot.PhysicsState();
        state.gravityScale = Physics.gravity.y;
        state.timeScale = Time.timeScale;
        state.useGravity = true; // Always using gravity in Unity

        states.Add(state);
        return states;
    }

    bool ShouldIncludeInSnapshot(GameObject obj)
    {
        // Define which objects should be included in snapshots
        // Exclude temporary objects, UI elements, etc.
        string[] excludedTags = { "Untagged", "UI", "Player", "MainCamera" };
        return !System.Array.Exists(excludedTags, tag => obj.tag == tag);
    }

    string GetPrefabName(GameObject obj)
    {
        // Get the prefab name for the object
        // This implementation depends on your prefab management system
        return obj.name;
    }

    void SaveSnapshot(EnvironmentSnapshot snapshot)
    {
        string path = Path.Combine(Application.persistentDataPath, snapshotDirectory,
            snapshot.environmentName + ".envsnap");

        BinaryFormatter formatter = new BinaryFormatter();
        using (FileStream stream = new FileStream(path, FileMode.Create))
        {
            formatter.Serialize(stream, snapshot);
        }

        Debug.Log($"Environment snapshot saved to: {path}");
    }

    public EnvironmentSnapshot LoadSnapshot(string name)
    {
        string path = Path.Combine(Application.persistentDataPath, snapshotDirectory, name + ".envsnap");

        if (!File.Exists(path))
        {
            Debug.LogError($"Snapshot file does not exist: {path}");
            return null;
        }

        BinaryFormatter formatter = new BinaryFormatter();
        using (FileStream stream = new FileStream(path, FileMode.Open))
        {
            EnvironmentSnapshot snapshot = (EnvironmentSnapshot)formatter.Deserialize(stream);
            return snapshot;
        }
    }

    public void ApplySnapshot(EnvironmentSnapshot snapshot)
    {
        if (snapshot == null) return;

        // Clear current environment
        ClearEnvironment();

        // Apply object states
        ApplyObjectStates(snapshot.objectStates);

        // Apply light states
        ApplyLightStates(snapshot.lightStates);

        // Apply physics states
        ApplyPhysicsStates(snapshot.physicsStates);

        Debug.Log($"Environment snapshot applied: {snapshot.environmentName}");
    }

    void ClearEnvironment()
    {
        // Clear the current environment (be careful with this!)
        // Only destroy objects that were created dynamically
        GameObject[] allObjects = FindObjectsOfType<GameObject>();

        foreach (GameObject obj in allObjects)
        {
            if (obj.transform.parent == null && ShouldIncludeInSnapshot(obj))
            {
                DestroyImmediate(obj);
            }
        }
    }

    void ApplyObjectStates(List<EnvironmentSnapshot.ObjectState> states)
    {
        foreach (EnvironmentSnapshot.ObjectState state in states)
        {
            GameObject obj = new GameObject(state.name);
            obj.transform.position = state.position;
            obj.transform.rotation = state.rotation;
            obj.transform.localScale = state.scale;
            obj.SetActive(state.isActive);

            // If this was a prefab, instantiate the appropriate prefab
            if (!string.IsNullOrEmpty(state.prefabName))
            {
                // Load and instantiate the prefab
                // This depends on your asset management system
            }
        }
    }

    void ApplyLightStates(List<EnvironmentSnapshot.LightState> states)
    {
        foreach (EnvironmentSnapshot.LightState state in states)
        {
            GameObject lightObj = new GameObject(state.name);
            Light light = lightObj.AddComponent<Light>();

            lightObj.transform.position = state.position;
            lightObj.transform.rotation = state.rotation;
            light.color = state.color;
            light.intensity = state.intensity;
            light.type = state.type;
        }
    }

    void ApplyPhysicsStates(List<EnvironmentSnapshot.PhysicsState> states)
    {
        if (states.Count > 0)
        {
            EnvironmentSnapshot.PhysicsState state = states[0];
            Physics.gravity = new Vector3(0, state.gravityScale, 0);
            Time.timeScale = state.timeScale;
        }
    }
}
```

## Environment Validation and Quality Assurance

### Simulation Fidelity Assessment

```csharp
using UnityEngine;
using System.Collections.Generic;

public class EnvironmentValidator : MonoBehaviour
{
    [Header("Validation Parameters")]
    public float positionTolerance = 0.01f;
    public float orientationTolerance = 1.0f; // degrees
    public float physicsTolerance = 0.1f;

    [Header("Validation Metrics")]
    public bool validatePhysics = true;
    public bool validateVisuals = true;
    public bool validateNavigation = true;
    public bool validateInteractions = true;

    [Header("Reference Data")]
    public GameObject referenceEnvironment;
    public List<ValidationPoint> validationPoints;

    [System.Serializable]
    public class ValidationPoint
    {
        public string name;
        public Vector3 position;
        public float radius;
        public ValidationType type;
    }

    public enum ValidationType
    {
        Physics,
        Visual,
        Navigation,
        Interaction
    }

    [System.Serializable]
    public class ValidationResult
    {
        public string testName;
        public bool passed;
        public float errorValue;
        public string errorMessage;
    }

    public List<ValidationResult> validationResults = new List<ValidationResult>();

    public void RunValidation()
    {
        validationResults.Clear();

        if (validatePhysics)
            ValidatePhysics();

        if (validateVisuals)
            ValidateVisuals();

        if (validateNavigation)
            ValidateNavigation();

        if (validateInteractions)
            ValidateInteractions();

        ReportValidationResults();
    }

    void ValidatePhysics()
    {
        // Compare physics behavior with reference
        ValidationResult result = new ValidationResult();
        result.testName = "Physics Validation";

        // Test object falling behavior
        float testObjectFallTime = TestObjectFallTime();
        float referenceFallTime = GetReferenceFallTime();

        result.errorValue = Mathf.Abs(testObjectFallTime - referenceFallTime);
        result.passed = result.errorValue <= physicsTolerance;
        result.errorMessage = result.passed ?
            "Physics behavior matches reference" :
            $"Physics error: {result.errorValue:F3}s difference";

        validationResults.Add(result);
    }

    float TestObjectFallTime()
    {
        // Drop an object and measure fall time
        GameObject testObject = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        testObject.transform.position = new Vector3(0, 2, 0);

        Rigidbody rb = testObject.AddComponent<Rigidbody>();
        rb.useGravity = true;

        float startTime = Time.time;
        float fallTime = 0f;

        // Wait for object to hit ground (y < 0.1)
        StartCoroutine(WaitForFall(testObject, ref fallTime, startTime));

        return fallTime;
    }

    System.Collections.IEnumerator WaitForFall(GameObject obj, ref float fallTime, float startTime)
    {
        while (obj.transform.position.y > 0.1f)
        {
            yield return null;
        }

        fallTime = Time.time - startTime;
        Destroy(obj);
    }

    float GetReferenceFallTime()
    {
        // This would come from reference data
        return 0.64f; // ~0.64 seconds to fall 2m with g=9.8
    }

    void ValidateVisuals()
    {
        // Compare visual properties with reference
        ValidationResult result = new ValidationResult();
        result.testName = "Visual Validation";

        // Check lighting, textures, materials, etc.
        // This might involve image comparison techniques

        result.passed = true; // Simplified for example
        result.errorValue = 0f;
        result.errorMessage = "Visual properties validated";

        validationResults.Add(result);
    }

    void ValidateNavigation()
    {
        // Test navigation mesh and pathfinding
        ValidationResult result = new ValidationResult();
        result.testName = "Navigation Validation";

        // Test pathfinding between validation points
        bool allPathsValid = true;
        float avgPathfindingTime = 0f;

        for (int i = 0; i < validationPoints.Count - 1; i++)
        {
            if (validationPoints[i].type == ValidationType.Navigation &&
                validationPoints[i + 1].type == ValidationType.Navigation)
            {
                // Test pathfinding between points
                if (!TestPathfinding(validationPoints[i].position, validationPoints[i + 1].position))
                {
                    allPathsValid = false;
                    break;
                }
            }
        }

        result.passed = allPathsValid;
        result.errorValue = allPathsValid ? 0f : 1f;
        result.errorMessage = allPathsValid ?
            "Navigation mesh is valid" :
            "Navigation mesh has issues";

        validationResults.Add(result);
    }

    bool TestPathfinding(Vector3 start, Vector3 end)
    {
        // Test if path exists between two points
        // This would use Unity's NavMesh system
        UnityEngine.AI.NavMeshPath path = new UnityEngine.AI.NavMeshPath();

        UnityEngine.AI.NavMesh.CalculatePath(start, end, UnityEngine.AI.NavMesh.AllAreas, path);

        return path.status == UnityEngine.AI.NavMeshPathStatus.PathComplete;
    }

    void ValidateInteractions()
    {
        // Test interactive object behavior
        ValidationResult result = new ValidationResult();
        result.testName = "Interaction Validation";

        // Test that interactive objects respond appropriately
        int interactiveObjects = FindObjectsOfType<ObjectInteraction>().Length;
        result.passed = interactiveObjects > 0;
        result.errorValue = interactiveObjects > 0 ? 0f : 1f;
        result.errorMessage = interactiveObjects > 0 ?
            $"{interactiveObjects} interactive objects found" :
            "No interactive objects found";

        validationResults.Add(result);
    }

    void ReportValidationResults()
    {
        Debug.Log("=== Environment Validation Report ===");
        int passedTests = 0;

        foreach (ValidationResult result in validationResults)
        {
            string status = result.passed ? "PASS" : "FAIL";
            Debug.Log($"{status}: {result.testName} - {result.errorMessage} (Error: {result.errorValue:F3})");

            if (result.passed) passedTests++;
        }

        Debug.Log($"Overall: {passedTests}/{validationResults.Count} tests passed");
    }

    public bool IsEnvironmentValid()
    {
        return validationResults.Count > 0 &&
               validationResults.FindAll(r => r.passed).Count == validationResults.Count;
    }
}
```

## Best Practices for Environment Design

### 1. Scalability and Reusability
- Design modular environments that can be combined
- Use parameterized configurations for flexibility
- Create template systems for common environment types

### 2. Performance Optimization
- Use appropriate level of detail for objects
- Implement occlusion culling for large environments
- Optimize physics calculations for interactive elements

### 3. Realism vs. Performance Balance
- Prioritize realism for critical test scenarios
- Use simplified models for performance-intensive tests
- Validate that simplifications don't affect test validity

### 4. Documentation and Version Control
- Document environment configurations and parameters
- Use version control for environment assets
- Maintain clear naming conventions

## Summary

This chapter covered environment and scenario building for humanoid robotics simulation, including procedural generation, dynamic environments, scenario design, and validation techniques. Well-designed environments are crucial for comprehensive testing of humanoid robot capabilities and for bridging the sim-to-real gap.

## Exercises

1. Create a home environment with kitchen, living room, and bedroom
2. Implement a procedural environment generator for office settings
3. Design a scenario that tests navigation and manipulation capabilities
4. Create an environment validation system for physics properties
5. Build a dynamic environment with moving obstacles and changing conditions

## Further Reading

- "Computer Graphics: Principles and Practice" for environment rendering
- "AI Game Programming Wisdom" for environment design principles
- Unity's ProBuilder documentation for rapid environment creation

---

*Next: [Module 3 Preface](../preface.md)*