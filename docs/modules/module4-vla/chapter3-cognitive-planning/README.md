# Chapter 3: Cognitive Planning Using LLMs

## Learning Objectives
By the end of this chapter, you will be able to:
- Understand how Large Language Models (LLMs) enable cognitive planning in robotics
- Design planning architectures that leverage LLM capabilities
- Implement multi-step task decomposition using LLMs
- Integrate LLM-based planning with robotic action execution
- Handle uncertainty and adapt planning based on environmental feedback
- Evaluate and validate LLM-based cognitive planning systems

## 3.1 Introduction to LLM-Based Cognitive Planning

Cognitive planning in robotics involves high-level reasoning about tasks, goals, and the environment to generate sequences of actions that achieve desired outcomes. Large Language Models (LLMs) have emerged as powerful tools for cognitive planning due to their ability to understand natural language, reason about complex scenarios, and generate structured plans.

### 3.1.1 The Role of LLMs in Cognitive Planning

Traditional robotic planning systems rely on symbolic representations and predefined rules or optimization algorithms. While effective for well-structured environments, these approaches struggle with complex, ambiguous, or novel situations that require common-sense reasoning and contextual understanding.

LLMs offer several advantages for cognitive planning:

- **Natural Language Understanding**: LLMs can directly process natural language instructions without requiring formal specification
- **Common-Sense Reasoning**: LLMs possess world knowledge that can guide planning in novel situations
- **Flexibility**: LLMs can adapt to new tasks and environments without reprogramming
- **Multi-step Reasoning**: LLMs can generate complex plans involving multiple subtasks
- **Context Awareness**: LLMs can consider environmental context and constraints in planning

### 3.1.2 Cognitive Planning Architecture

The integration of LLMs with robotic planning creates a hybrid architecture:

```
[Perception] → [Environment State] → [LLM Planner] → [Action Sequence]
     ↓              ↓                    ↓              ↓
[Robot State] → [Goal Specification] → [Plan Refinement] → [Execution]
```

## 3.2 LLM Integration for Planning

### 3.2.1 Planning Prompts and Templates

Effective cognitive planning requires well-crafted prompts that guide the LLM to generate appropriate plans:

```python
class PlanningPromptTemplate:
    """Template for generating planning prompts for LLMs"""

    def __init__(self):
        self.system_prompt = """
        You are a cognitive planning assistant for a humanoid robot. Your role is to generate detailed, step-by-step plans for completing tasks in a physical environment.

        Requirements:
        1. Generate plans that are feasible for a humanoid robot
        2. Include specific actions with object and location references
        3. Consider environmental constraints and safety
        4. Handle uncertainty by including verification steps
        5. Break complex tasks into manageable subtasks
        6. Include error handling and recovery steps
        """

    def create_task_planning_prompt(self, task_description, environment_state, robot_capabilities):
        """
        Create a prompt for task planning based on the current context

        Args:
            task_description: Natural language description of the task
            environment_state: Current state of the environment
            robot_capabilities: Capabilities of the robot

        Returns:
            Formatted prompt for LLM
        """
        prompt = f"""
        {self.system_prompt}

        TASK: {task_description}

        ENVIRONMENT STATE:
        {environment_state}

        ROBOT CAPABILITIES:
        {robot_capabilities}

        Generate a detailed plan to complete the task. Format your response as:
        1. Plan Overview: Brief summary of the approach
        2. Prerequisites: What needs to be verified before starting
        3. Step-by-Step Plan: Detailed actions with expected outcomes
        4. Success Criteria: How to verify task completion
        5. Potential Issues: What could go wrong and how to handle it

        Be specific about objects, locations, and actions. Use only capabilities available to the robot.
        """

        return prompt

    def create_replanning_prompt(self, original_plan, current_state, failure_point):
        """
        Create a prompt for replanning when the original plan fails

        Args:
            original_plan: The plan that failed
            current_state: Current state after failure
            failure_point: Where and why the plan failed

        Returns:
            Formatted prompt for replanning
        """
        prompt = f"""
        {self.system_prompt}

        ORIGINAL PLAN:
        {original_plan}

        CURRENT STATE AFTER FAILURE:
        {current_state}

        FAILURE POINT: {failure_point}

        Generate a revised plan that addresses the failure and completes the original task.
        Consider alternative approaches and adapt to the current state.
        """

        return prompt
```

### 3.2.2 Environment State Representation

LLMs need structured information about the environment to generate effective plans:

```python
class EnvironmentState:
    """Represents the current state of the robot's environment"""

    def __init__(self):
        self.objects = {}  # {object_id: object_info}
        self.locations = {}  # {location_id: location_info}
        self.robot_state = {}  # Robot's current state
        self.constraints = []  # Environmental constraints
        self.time = None  # Current time
        self.weather = None  # Environmental conditions (if relevant)

    def to_text_description(self):
        """Convert environment state to text description for LLM"""
        description = "ENVIRONMENT STATE:\n"

        # Objects in environment
        description += "Objects:\n"
        for obj_id, obj_info in self.objects.items():
            description += f"  - {obj_id}: {obj_info['type']} at {obj_info['location']}"
            if 'properties' in obj_info:
                properties = ', '.join([f"{k}={v}" for k, v in obj_info['properties'].items()])
                description += f" ({properties})"
            description += "\n"

        # Locations
        description += "\nLocations:\n"
        for loc_id, loc_info in self.locations.items():
            description += f"  - {loc_id}: {loc_info['type']} - {loc_info['description']}\n"
            if 'objects' in loc_info:
                description += f"    Contains: {', '.join(loc_info['objects'])}\n"

        # Robot state
        description += f"\nRobot State:\n"
        description += f"  - Position: {self.robot_state.get('position', 'unknown')}\n"
        description += f"  - Battery: {self.robot_state.get('battery', 'unknown')}\n"
        description += f"  - Available tools: {self.robot_state.get('tools', [])}\n"

        # Constraints
        if self.constraints:
            description += f"\nConstraints:\n"
            for constraint in self.constraints:
                description += f"  - {constraint}\n"

        return description

    def update_object(self, obj_id, obj_info):
        """Update information about a specific object"""
        self.objects[obj_id] = obj_info

    def update_location(self, loc_id, loc_info):
        """Update information about a specific location"""
        self.locations[loc_id] = loc_info

    def add_constraint(self, constraint):
        """Add an environmental constraint"""
        self.constraints.append(constraint)

    def remove_constraint(self, constraint):
        """Remove an environmental constraint"""
        if constraint in self.constraints:
            self.constraints.remove(constraint)
```

### 3.2.3 Robot Capability Modeling

LLMs need to understand what the robot can and cannot do:

```python
class RobotCapabilities:
    """Represents the capabilities of a specific robot"""

    def __init__(self, robot_model="default"):
        self.model = robot_model
        self.capabilities = self._define_capabilities()
        self.limits = self._define_limits()
        self.tools = self._define_tools()

    def _define_capabilities(self):
        """Define what the robot can do"""
        return {
            "navigation": {
                "type": "locomotion",
                "description": "Move to specified locations",
                "parameters": ["target_location", "speed", "path_preference"]
            },
            "manipulation": {
                "type": "manipulation",
                "description": "Pick up, place, and manipulate objects",
                "parameters": ["object", "target_location", "gripper_force"]
            },
            "perception": {
                "type": "sensing",
                "description": "Detect and identify objects and people",
                "parameters": ["object_type", "detection_range"]
            },
            "communication": {
                "type": "interaction",
                "description": "Speak, listen, and interact with humans",
                "parameters": ["message", "recipient"]
            },
            "grasping": {
                "type": "manipulation",
                "description": "Grasp objects with specified grip type",
                "parameters": ["object", "grip_type", "force"]
            },
            "transport": {
                "type": "manipulation",
                "description": "Carry objects from one location to another",
                "parameters": ["object", "source", "destination"]
            }
        }

    def _define_limits(self):
        """Define the limits of robot capabilities"""
        return {
            "max_payload": 5.0,  # kg
            "max_reach": 1.5,    # meters
            "max_speed": 1.0,    # m/s
            "battery_life": 8.0, # hours
            "precision": 0.01,   # meters
            "object_size_min": 0.01,  # meters
            "object_size_max": 0.5    # meters
        }

    def _define_tools(self):
        """Define tools and end-effectors available"""
        return {
            "gripper": {
                "type": "parallel_gripper",
                "max_force": 100.0,  # Newtons
                "max_opening": 0.1,  # meters
                "precision": "high"
            },
            "camera": {
                "type": "rgb_depth",
                "resolution": "1920x1080",
                "fov": 60.0  # degrees
            }
        }

    def to_text_description(self):
        """Convert capabilities to text for LLM"""
        description = "ROBOT CAPABILITIES:\n"

        description += "Available Actions:\n"
        for action_name, action_info in self.capabilities.items():
            description += f"  - {action_name}: {action_info['description']}\n"
            description += f"    Parameters: {', '.join(action_info['parameters'])}\n"

        description += "\nPhysical Limits:\n"
        for limit_name, limit_value in self.limits.items():
            description += f"  - {limit_name}: {limit_value}\n"

        description += "\nTools Available:\n"
        for tool_name, tool_info in self.tools.items():
            description += f"  - {tool_name}: {tool_info['type']}\n"

        return description

    def can_perform_action(self, action_name, parameters=None):
        """Check if robot can perform a specific action"""
        if action_name not in self.capabilities:
            return False

        # Additional checks could be added here based on parameters
        return True
```

## 3.3 Plan Generation and Structuring

### 3.3.1 Plan Structure and Validation

Generating structured plans that can be executed by the robot:

```python
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class PlanStepType(Enum):
    NAVIGATION = "navigation"
    MANIPULATION = "manipulation"
    PERCEPTION = "perception"
    COMMUNICATION = "communication"
    WAIT = "wait"
    CONDITIONAL = "conditional"

@dataclass
class PlanStep:
    """Represents a single step in a robot plan"""
    step_id: str
    action_type: PlanStepType
    description: str
    parameters: Dict[str, Any]
    expected_duration: float  # seconds
    preconditions: List[str]  # Conditions that must be true
    postconditions: List[str]  # Conditions that will be true after execution
    success_criteria: List[str]  # How to verify success
    failure_recovery: Optional[str]  # What to do if step fails

class Plan:
    """Represents a complete plan for task execution"""

    def __init__(self, task_description: str, steps: List[PlanStep]):
        self.task_description = task_description
        self.steps = steps
        self.created_at = self._get_current_time()
        self.plan_id = self._generate_plan_id()

    def _get_current_time(self):
        """Get current timestamp"""
        import datetime
        return datetime.datetime.now().isoformat()

    def _generate_plan_id(self):
        """Generate unique plan ID"""
        import uuid
        return str(uuid.uuid4())

    def to_dict(self):
        """Convert plan to dictionary for serialization"""
        return {
            "plan_id": self.plan_id,
            "task_description": self.task_description,
            "created_at": self.created_at,
            "steps": [
                {
                    "step_id": step.step_id,
                    "action_type": step.action_type.value,
                    "description": step.description,
                    "parameters": step.parameters,
                    "expected_duration": step.expected_duration,
                    "preconditions": step.preconditions,
                    "postconditions": step.postconditions,
                    "success_criteria": step.success_criteria,
                    "failure_recovery": step.failure_recovery
                }
                for step in self.steps
            ]
        }

    def to_json(self):
        """Convert plan to JSON string"""
        return json.dumps(self.to_dict(), indent=2)

    def validate(self) -> tuple[bool, List[str]]:
        """Validate the plan for consistency and feasibility"""
        errors = []

        # Check for missing preconditions
        available_conditions = set()
        for i, step in enumerate(self.steps):
            # Check if preconditions are satisfied
            for precondition in step.preconditions:
                if precondition not in available_conditions:
                    errors.append(f"Step {i+1} ({step.description}) requires precondition '{precondition}' that is not available")

            # Add postconditions to available conditions
            for postcondition in step.postconditions:
                available_conditions.add(postcondition)

        # Check for valid action types
        valid_action_types = [e.value for e in PlanStepType]
        for i, step in enumerate(self.steps):
            if step.action_type.value not in valid_action_types:
                errors.append(f"Step {i+1} has invalid action type: {step.action_type.value}")

        return len(errors) == 0, errors
```

### 3.3.2 LLM Plan Generation

Generating plans using LLMs with structured output:

```python
import openai
import json
import re
from typing import Optional

class LLMPlanGenerator:
    """Generates plans using Large Language Models"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Initialize LLM plan generator

        Args:
            api_key: OpenAI API key
            model: LLM model to use
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.prompt_template = PlanningPromptTemplate()

    def generate_plan(self, task_description: str, environment_state: EnvironmentState,
                     robot_capabilities: RobotCapabilities) -> Optional[Plan]:
        """
        Generate a plan for the given task using LLM

        Args:
            task_description: Natural language description of the task
            environment_state: Current environment state
            robot_capabilities: Robot's capabilities

        Returns:
            Generated Plan object or None if generation failed
        """
        # Create prompt
        prompt = self.prompt_template.create_task_planning_prompt(
            task_description,
            environment_state.to_text_description(),
            robot_capabilities.to_text_description()
        )

        try:
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a cognitive planning assistant for a humanoid robot. Generate structured plans in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent output
                max_tokens=2000
            )

            # Parse the response
            plan_json = self._extract_json_from_response(response.choices[0].message.content)
            if plan_json:
                return self._parse_plan_from_json(plan_json, task_description)
            else:
                print("Could not extract valid JSON from LLM response")
                return None

        except Exception as e:
            print(f"Error generating plan: {e}")
            return None

    def _extract_json_from_response(self, response_text: str) -> Optional[Dict]:
        """Extract JSON from LLM response that may contain additional text"""
        # Look for JSON between ```json and ``` markers
        json_match = re.search(r'```json\s*\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Look for JSON between {} brackets
            bracket_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if bracket_match:
                json_str = bracket_match.group(0)
            else:
                return None

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"Could not parse JSON: {json_str}")
            return None

    def _parse_plan_from_json(self, plan_json: Dict, task_description: str) -> Optional[Plan]:
        """Parse plan from JSON structure to Plan object"""
        try:
            steps = []
            for step_data in plan_json.get("steps", []):
                step = PlanStep(
                    step_id=step_data.get("step_id", f"step_{len(steps)+1}"),
                    action_type=PlanStepType(step_data.get("action_type", "navigation")),
                    description=step_data.get("description", ""),
                    parameters=step_data.get("parameters", {}),
                    expected_duration=step_data.get("expected_duration", 10.0),
                    preconditions=step_data.get("preconditions", []),
                    postconditions=step_data.get("postconditions", []),
                    success_criteria=step_data.get("success_criteria", []),
                    failure_recovery=step_data.get("failure_recovery")
                )
                steps.append(step)

            plan = Plan(task_description, steps)

            # Validate the plan
            is_valid, errors = plan.validate()
            if not is_valid:
                print(f"Generated plan has validation errors: {errors}")
                return None

            return plan

        except Exception as e:
            print(f"Error parsing plan from JSON: {e}")
            return None

    def refine_plan(self, original_plan: Plan, feedback: str) -> Optional[Plan]:
        """Refine an existing plan based on feedback"""
        prompt = f"""
        Refine the following plan based on the provided feedback:

        ORIGINAL PLAN:
        {original_plan.to_json()}

        FEEDBACK:
        {feedback}

        Return the refined plan in the same JSON format.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a cognitive planning assistant. Refine plans based on feedback."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )

            plan_json = self._extract_json_from_response(response.choices[0].message.content)
            if plan_json:
                return self._parse_plan_from_json(plan_json, original_plan.task_description)
            else:
                return original_plan  # Return original if refinement failed

        except Exception as e:
            print(f"Error refining plan: {e}")
            return original_plan
```

## 3.4 Plan Execution and Monitoring

### 3.4.1 Plan Execution Framework

Executing plans while monitoring progress and handling deviations:

```python
import time
import threading
from typing import Callable, Any
from enum import Enum

class ExecutionStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"

class PlanExecutor:
    """Executes plans generated by LLM with monitoring and recovery"""

    def __init__(self, robot_interface):
        """
        Initialize plan executor

        Args:
            robot_interface: Interface to the physical or simulated robot
        """
        self.robot_interface = robot_interface
        self.current_plan = None
        self.current_status = ExecutionStatus.PENDING
        self.current_step_index = 0
        self.execution_thread = None
        self.should_stop = False

    def execute_plan(self, plan: Plan, on_status_change: Callable[[ExecutionStatus], None] = None) -> bool:
        """
        Execute a plan with monitoring

        Args:
            plan: Plan to execute
            on_status_change: Callback for status changes

        Returns:
            True if plan completed successfully, False otherwise
        """
        self.current_plan = plan
        self.current_status = ExecutionStatus.EXECUTING
        self.current_step_index = 0
        self.should_stop = False

        if on_status_change:
            on_status_change(self.current_status)

        # Execute in separate thread to allow monitoring
        self.execution_thread = threading.Thread(
            target=self._execute_plan_thread,
            args=(on_status_change,)
        )
        self.execution_thread.start()

        # Wait for completion
        self.execution_thread.join()

        return self.current_status == ExecutionStatus.SUCCESS

    def _execute_plan_thread(self, on_status_change: Callable[[ExecutionStatus], None]):
        """Execute plan in a separate thread"""
        try:
            while (self.current_step_index < len(self.current_plan.steps) and
                   not self.should_stop and
                   self.current_status == ExecutionStatus.EXECUTING):

                step = self.current_plan.steps[self.current_step_index]

                # Check preconditions
                if not self._check_preconditions(step):
                    self.current_status = ExecutionStatus.FAILED
                    if on_status_change:
                        on_status_change(self.current_status)
                    break

                # Execute the step
                success = self._execute_step(step)

                if success:
                    # Verify postconditions
                    if not self._verify_postconditions(step):
                        self.current_status = ExecutionStatus.FAILED
                        if on_status_change:
                            on_status_change(self.current_status)
                        break

                    self.current_step_index += 1
                else:
                    # Step failed, try recovery
                    if step.failure_recovery:
                        recovered = self._attempt_recovery(step)
                        if recovered:
                            continue

                    self.current_status = ExecutionStatus.FAILED
                    if on_status_change:
                        on_status_change(self.current_status)
                    break

            # Check if all steps completed
            if (self.current_status == ExecutionStatus.EXECUTING and
                self.current_step_index >= len(self.current_plan.steps)):
                self.current_status = ExecutionStatus.SUCCESS
                if on_status_change:
                    on_status_change(self.current_status)

        except Exception as e:
            print(f"Error during plan execution: {e}")
            self.current_status = ExecutionStatus.FAILED
            if on_status_change:
                on_status_change(self.current_status)

    def _check_preconditions(self, step: PlanStep) -> bool:
        """Check if preconditions for a step are met"""
        for precondition in step.preconditions:
            if not self._evaluate_condition(precondition):
                print(f"Precondition not met: {precondition}")
                return False
        return True

    def _verify_postconditions(self, step: PlanStep) -> bool:
        """Verify that postconditions are satisfied after step execution"""
        for postcondition in step.postconditions:
            if not self._evaluate_condition(postcondition):
                print(f"Postcondition not satisfied: {postcondition}")
                return False
        return True

    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate a condition against current robot state"""
        # This would interface with robot's sensors and state
        # For now, we'll implement a simple rule-based evaluation

        # Example conditions: "object_at_location(cup, kitchen_table)"
        if condition.startswith("object_at_location"):
            # Parse: object_at_location(object, location)
            import re
            match = re.search(r'object_at_location\(([^,]+),\s*([^)]+)\)', condition)
            if match:
                obj, location = match.groups()
                # Check if robot can perceive the object at the location
                return self.robot_interface.is_object_at_location(obj.strip(), location.strip())

        elif condition.startswith("robot_at_location"):
            import re
            match = re.search(r'robot_at_location\(([^)]+)\)', condition)
            if match:
                location = match.group(1)
                return self.robot_interface.get_robot_location() == location.strip()

        elif condition.startswith("object_grasped"):
            import re
            match = re.search(r'object_grasped\(([^)]+)\)', condition)
            if match:
                obj = match.group(1)
                return self.robot_interface.is_object_grasped(obj.strip())

        # Default: assume condition is true if not specifically handled
        return True

    def _execute_step(self, step: PlanStep) -> bool:
        """Execute a single plan step"""
        print(f"Executing step: {step.description}")

        try:
            if step.action_type == PlanStepType.NAVIGATION:
                return self._execute_navigation_step(step)
            elif step.action_type == PlanStepType.MANIPULATION:
                return self._execute_manipulation_step(step)
            elif step.action_type == PlanStepType.PERCEPTION:
                return self._execute_perception_step(step)
            elif step.action_type == PlanStepType.COMMUNICATION:
                return self._execute_communication_step(step)
            elif step.action_type == PlanStepType.WAIT:
                return self._execute_wait_step(step)
            else:
                print(f"Unknown action type: {step.action_type}")
                return False

        except Exception as e:
            print(f"Error executing step {step.step_id}: {e}")
            return False

    def _execute_navigation_step(self, step: PlanStep) -> bool:
        """Execute a navigation step"""
        target_location = step.parameters.get("target_location")
        if not target_location:
            print("Navigation step missing target location")
            return False

        # Execute navigation through robot interface
        success = self.robot_interface.navigate_to_location(target_location)

        # Wait for completion or timeout
        timeout = step.expected_duration
        start_time = time.time()

        while (time.time() - start_time < timeout and
               self.robot_interface.is_navigation_active()):
            time.sleep(0.1)  # Check status periodically

        return success

    def _execute_manipulation_step(self, step: PlanStep) -> bool:
        """Execute a manipulation step"""
        action = step.parameters.get("action")

        if action == "pick":
            object_name = step.parameters.get("object")
            location = step.parameters.get("location")
            return self.robot_interface.pick_object(object_name, location)
        elif action == "place":
            object_name = step.parameters.get("object")
            target_location = step.parameters.get("target_location")
            return self.robot_interface.place_object(object_name, target_location)
        elif action == "grasp":
            object_name = step.parameters.get("object")
            grip_type = step.parameters.get("grip_type", "default")
            return self.robot_interface.grasp_object(object_name, grip_type)
        else:
            print(f"Unknown manipulation action: {action}")
            return False

    def _execute_perception_step(self, step: PlanStep) -> bool:
        """Execute a perception step"""
        object_type = step.parameters.get("object_type")
        detection_range = step.parameters.get("detection_range", 3.0)

        # Perform perception through robot interface
        detected_objects = self.robot_interface.detect_objects(
            object_type=object_type,
            max_range=detection_range
        )

        # Store results for later steps
        self.robot_interface.store_perception_results(detected_objects)

        return len(detected_objects) > 0

    def _execute_communication_step(self, step: PlanStep) -> bool:
        """Execute a communication step"""
        message = step.parameters.get("message")
        recipient = step.parameters.get("recipient", "user")

        return self.robot_interface.speak(message)

    def _execute_wait_step(self, step: PlanStep) -> bool:
        """Execute a wait step"""
        duration = step.parameters.get("duration", 1.0)
        time.sleep(duration)
        return True

    def _attempt_recovery(self, failed_step: PlanStep) -> bool:
        """Attempt to recover from a failed step"""
        print(f"Attempting recovery for failed step: {failed_step.description}")

        if failed_step.failure_recovery:
            # Parse and execute recovery action
            recovery_parts = failed_step.failure_recovery.split('(')
            if len(recovery_parts) >= 2:
                recovery_action = recovery_parts[0]
                recovery_params_str = recovery_parts[1].rstrip(')')

                # Simple recovery actions
                if recovery_action == "retry":
                    print("Retrying failed step...")
                    return self._execute_step(failed_step)
                elif recovery_action == "skip":
                    print("Skipping failed step...")
                    self.current_step_index += 1
                    return True
                elif recovery_action == "replan":
                    print("Replanning needed...")
                    return False  # Indicate that replanning is needed

        return False

    def cancel_execution(self):
        """Cancel current plan execution"""
        self.should_stop = True
        self.current_status = ExecutionStatus.CANCELLED

    def pause_execution(self):
        """Pause current plan execution"""
        self.current_status = ExecutionStatus.PAUSED

    def resume_execution(self):
        """Resume paused plan execution"""
        if self.current_status == ExecutionStatus.PAUSED:
            self.current_status = ExecutionStatus.EXECUTING
```

### 3.4.2 Plan Monitoring and Adaptation

Monitoring plan execution and adapting to environmental changes:

```python
import threading
import time
from typing import Dict, List, Callable

class PlanMonitor:
    """Monitors plan execution and detects deviations"""

    def __init__(self, robot_interface, plan_executor):
        """
        Initialize plan monitor

        Args:
            robot_interface: Interface to the robot
            plan_executor: Plan executor to monitor
        """
        self.robot_interface = robot_interface
        self.plan_executor = plan_executor
        self.monitoring_thread = None
        self.is_monitoring = False
        self.deviation_callbacks = []
        self.environment_change_callbacks = []

    def start_monitoring(self):
        """Start monitoring plan execution"""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()

    def add_deviation_callback(self, callback: Callable[[str, Dict], None]):
        """Add callback for plan deviations"""
        self.deviation_callbacks.append(callback)

    def add_environment_change_callback(self, callback: Callable[[Dict], None]):
        """Add callback for environment changes"""
        self.environment_change_callbacks.append(callback)

    def _monitoring_loop(self):
        """Main monitoring loop"""
        last_environment_state = self._get_current_environment_state()

        while self.is_monitoring:
            try:
                # Check if plan is still executing
                if self.plan_executor.current_status != ExecutionStatus.EXECUTING:
                    time.sleep(1.0)
                    continue

                # Get current state
                current_environment_state = self._get_current_environment_state()

                # Check for environment changes
                changes = self._detect_environment_changes(
                    last_environment_state, current_environment_state
                )

                if changes:
                    # Notify callbacks about environment changes
                    for callback in self.environment_change_callbacks:
                        callback(changes)

                    # Check if changes affect current plan
                    if self._changes_affect_plan(changes):
                        # Consider replanning
                        self._handle_environment_changes(changes)

                # Check for plan deviations
                current_step = self.plan_executor.current_plan.steps[
                    self.plan_executor.current_step_index
                ] if self.plan_executor.current_step_index < len(self.plan_executor.current_plan.steps) else None

                if current_step:
                    deviation = self._check_for_deviation(current_step)
                    if deviation:
                        for callback in self.deviation_callbacks:
                            callback(deviation, current_step)

                last_environment_state = current_environment_state
                time.sleep(0.5)  # Check every 500ms

            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(1.0)

    def _get_current_environment_state(self) -> Dict:
        """Get current environment state from robot sensors"""
        return {
            "objects": self.robot_interface.get_visible_objects(),
            "robot_location": self.robot_interface.get_robot_location(),
            "robot_battery": self.robot_interface.get_battery_level(),
            "obstacles": self.robot_interface.get_detected_obstacles(),
            "time": time.time()
        }

    def _detect_environment_changes(self, old_state: Dict, new_state: Dict) -> Dict:
        """Detect changes between environment states"""
        changes = {}

        # Check for object changes
        old_objects = set(old_state.get("objects", []))
        new_objects = set(new_state.get("objects", []))

        if old_objects != new_objects:
            changes["objects"] = {
                "added": list(new_objects - old_objects),
                "removed": list(old_objects - new_objects),
                "changed": []  # Could add more detailed object state changes
            }

        # Check for location changes
        if old_state.get("robot_location") != new_state.get("robot_location"):
            changes["robot_location"] = {
                "from": old_state["robot_location"],
                "to": new_state["robot_location"]
            }

        # Check for obstacle changes
        old_obstacles = set(old_state.get("obstacles", []))
        new_obstacles = set(new_state.get("obstacles", []))

        if old_obstacles != new_obstacles:
            changes["obstacles"] = {
                "added": list(new_obstacles - old_obstacles),
                "removed": list(old_obstacles - new_obstacles)
            }

        return changes

    def _changes_affect_plan(self, changes: Dict) -> bool:
        """Check if environment changes affect the current plan"""
        current_plan = self.plan_executor.current_plan
        current_step_idx = self.plan_executor.current_step_index

        if not current_plan or current_step_idx >= len(current_plan.steps):
            return False

        # Check if obstacles affect navigation steps
        if "obstacles" in changes and changes["obstacles"]["added"]:
            for i in range(current_step_idx, len(current_plan.steps)):
                step = current_plan.steps[i]
                if (step.action_type == PlanStepType.NAVIGATION and
                    self._obstacle_blocks_path(changes["obstacles"]["added"], step)):
                    return True

        # Check if object changes affect manipulation steps
        if "objects" in changes:
            for i in range(current_step_idx, len(current_plan.steps)):
                step = current_plan.steps[i]
                if (step.action_type == PlanStepType.MANIPULATION and
                    step.parameters.get("object") in changes["objects"]["removed"]):
                    return True

        return False

    def _obstacle_blocks_path(self, obstacles: List, step: PlanStep) -> bool:
        """Check if obstacles block the path for a navigation step"""
        # This would require path planning integration
        # For now, assume any new obstacle in the general direction is blocking
        target_location = step.parameters.get("target_location")
        robot_location = self.robot_interface.get_robot_location()

        # Simple check: if obstacle is between robot and target
        for obstacle in obstacles:
            # This is a simplified check - in practice would need proper path planning
            if self._is_obstacle_in_path(robot_location, target_location, obstacle):
                return True

        return False

    def _is_obstacle_in_path(self, start, end, obstacle) -> bool:
        """Check if obstacle is in the path between start and end"""
        # Simplified implementation - would need proper geometric calculations
        return True  # Conservative assumption

    def _check_for_deviation(self, current_step: PlanStep) -> str:
        """Check if execution is deviating from expected behavior"""
        # Check if step is taking too long
        if hasattr(current_step, '_start_time'):
            elapsed = time.time() - current_step._start_time
            if elapsed > current_step.expected_duration * 2:  # 2x expected time
                return f"Step taking too long: expected {current_step.expected_duration}s, running for {elapsed:.1f}s"

        # Check for unexpected robot states
        robot_state = self.robot_interface.get_robot_state()
        if robot_state.get("battery_level", 100) < 10:  # Low battery
            return "Robot battery critically low"

        # Check for unexpected obstacles
        obstacles = self.robot_interface.get_detected_obstacles()
        if len(obstacles) > 5:  # Too many unexpected obstacles
            return "Unexpected obstacles detected"

        return None

    def _handle_environment_changes(self, changes: Dict):
        """Handle detected environment changes"""
        print(f"Environment changes detected: {changes}")

        # For now, just log the changes
        # In a real system, this might trigger replanning
        pass
```

## 3.5 Advanced Planning Concepts

### 3.5.1 Hierarchical Task Networks (HTN)

Implementing hierarchical planning for complex tasks:

```python
from typing import Union, List
from dataclasses import dataclass

@dataclass
class Task:
    """Represents a task in hierarchical planning"""
    name: str
    description: str
    subtasks: List['Task'] = None
    primitive_action: PlanStep = None
    preconditions: List[str] = None
    postconditions: List[str] = None

class HTNPlanner:
    """Hierarchical Task Network planner using LLMs"""

    def __init__(self, llm_generator: LLMPlanGenerator):
        self.llm_generator = llm_generator

    def decompose_task(self, task: Task, environment_state: EnvironmentState) -> List[PlanStep]:
        """
        Decompose a high-level task into primitive actions

        Args:
            task: High-level task to decompose
            environment_state: Current environment state

        Returns:
            List of primitive PlanStep objects
        """
        if task.primitive_action is not None:
            # Task is already primitive
            return [task.primitive_action]

        if task.subtasks is not None:
            # Decompose into subtasks
            steps = []
            for subtask in task.subtasks:
                subtask_steps = self.decompose_task(subtask, environment_state)
                steps.extend(subtask_steps)
            return steps

        # Use LLM to decompose the task
        return self._llm_decompose_task(task, environment_state)

    def _llm_decompose_task(self, task: Task, environment_state: EnvironmentState) -> List[PlanStep]:
        """Use LLM to decompose a task into subtasks or primitive actions"""
        prompt = f"""
        Decompose the following high-level task into specific, executable steps for a humanoid robot:

        TASK: {task.name}
        DESCRIPTION: {task.description}

        ENVIRONMENT STATE:
        {environment_state.to_text_description()}

        Decompose this task into 2-5 specific subtasks or primitive actions. Return as JSON with the following structure:
        {{
            "subtasks": [
                {{
                    "name": "subtask_name",
                    "description": "what to do",
                    "action_type": "navigation|manipulation|perception|communication",
                    "parameters": {{"param1": "value1", ...}},
                    "expected_duration": seconds,
                    "preconditions": ["condition1", "condition2"],
                    "postconditions": ["condition1", "condition2"],
                    "success_criteria": ["criterion1", "criterion2"]
                }}
            ]
        }}
        """

        try:
            response = self.llm_generator.client.chat.completions.create(
                model=self.llm_generator.model,
                messages=[
                    {"role": "system", "content": "You are a task decomposition expert. Break down high-level tasks into executable subtasks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )

            result_json = self.llm_generator._extract_json_from_response(response.choices[0].message.content)
            if result_json and "subtasks" in result_json:
                steps = []
                for subtask_data in result_json["subtasks"]:
                    step = PlanStep(
                        step_id=f"htn_{len(steps)+1}",
                        action_type=PlanStepType(subtask_data.get("action_type", "navigation")),
                        description=subtask_data.get("description", ""),
                        parameters=subtask_data.get("parameters", {}),
                        expected_duration=subtask_data.get("expected_duration", 10.0),
                        preconditions=subtask_data.get("preconditions", []),
                        postconditions=subtask_data.get("postconditions", []),
                        success_criteria=subtask_data.get("success_criteria", []),
                        failure_recovery=None
                    )
                    steps.append(step)

                return steps

        except Exception as e:
            print(f"Error decomposing task with LLM: {e}")
            # Return a default simple decomposition
            return [PlanStep(
                step_id="default_step",
                action_type=PlanStepType.NAVIGATION,
                description=f"Execute task: {task.name}",
                parameters={"task": task.name},
                expected_duration=30.0,
                preconditions=[],
                postconditions=[],
                success_criteria=[f"{task.name} completed"],
                failure_recovery="retry"
            )]

    def create_plan_from_task(self, task: Task, environment_state: EnvironmentState,
                             robot_capabilities: RobotCapabilities) -> Plan:
        """Create a complete plan from a high-level task"""
        primitive_steps = self.decompose_task(task, environment_state)
        return Plan(task.description, primitive_steps)
```

### 3.5.2 Uncertainty Handling and Probabilistic Planning

Handling uncertainty in LLM-based planning:

```python
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class UncertainCondition:
    """Represents a condition with associated probability"""
    condition: str
    probability: float  # 0.0 to 1.0
    confidence: float   # Confidence in the probability estimate

class UncertaintyHandler:
    """Handles uncertainty in LLM-based planning"""

    def __init__(self):
        self.uncertainty_threshold = 0.7  # Minimum probability for "likely" conditions

    def evaluate_uncertain_conditions(self, conditions: List[UncertainCondition]) -> Dict[str, bool]:
        """
        Evaluate uncertain conditions based on their probabilities

        Args:
            conditions: List of conditions with probabilities

        Returns:
            Dictionary mapping condition names to whether they're considered true
        """
        evaluations = {}

        for cond in conditions:
            # Consider condition true if probability exceeds threshold
            evaluations[cond.condition] = cond.probability >= self.uncertainty_threshold

        return evaluations

    def generate_contingency_plans(self, base_plan: Plan, uncertainties: List[UncertainCondition]) -> List[Plan]:
        """
        Generate contingency plans for different uncertainty outcomes

        Args:
            base_plan: Original plan with uncertainties
            uncertainties: List of uncertain conditions

        Returns:
            List of plans for different possible outcomes
        """
        contingency_plans = [base_plan]  # Start with base plan

        # For each uncertain condition, generate alternative plans
        for uncertainty in uncertainties:
            if uncertainty.probability < self.uncertainty_threshold:
                # Generate alternative plan for when condition is false
                alternative_plan = self._create_alternative_plan(base_plan, uncertainty)
                contingency_plans.append(alternative_plan)

        return contingency_plans

    def _create_alternative_plan(self, base_plan: Plan, uncertainty: UncertainCondition) -> Plan:
        """Create an alternative plan assuming the uncertainty is false"""
        # This would involve replanning with the uncertain condition negated
        # For now, we'll create a modified version of the base plan
        alternative_steps = []

        for step in base_plan.steps:
            # Check if this step depends on the uncertain condition
            if uncertainty.condition in step.preconditions:
                # Create alternative step that doesn't require the condition
                alt_step = PlanStep(
                    step_id=f"alt_{step.step_id}",
                    action_type=step.action_type,
                    description=f"Alternative to: {step.description}",
                    parameters=step.parameters.copy(),
                    expected_duration=step.expected_duration,
                    preconditions=[c for c in step.preconditions if c != uncertainty.condition],
                    postconditions=step.postconditions,
                    success_criteria=step.success_criteria,
                    failure_recovery=step.failure_recovery
                )
                alternative_steps.append(alt_step)
            else:
                alternative_steps.append(step)

        return Plan(f"Alternative plan for {base_plan.task_description}", alternative_steps)

    def update_condition_probabilities(self, condition: str, new_evidence: Dict) -> float:
        """
        Update probability of a condition based on new evidence using Bayesian updating

        Args:
            condition: The condition to update
            new_evidence: New evidence affecting the condition

        Returns:
            Updated probability
        """
        # This would implement Bayesian probability updating
        # For now, return a simple weighted average
        prior_probability = new_evidence.get('prior_probability', 0.5)
        evidence_strength = new_evidence.get('evidence_strength', 0.5)
        evidence_direction = new_evidence.get('evidence_direction', 0.0)  # -1 to 1

        # Update probability based on evidence
        updated_probability = prior_probability + evidence_strength * evidence_direction
        return max(0.0, min(1.0, updated_probability))  # Clamp to [0,1]
```

## 3.6 Integration with VLA Systems

### 3.6.1 Complete Cognitive Planning Pipeline

Integrating cognitive planning with the broader VLA system:

```python
class VLACognitivePlanner:
    """Complete cognitive planning system for VLA robots"""

    def __init__(self, robot_interface, api_key: str):
        """
        Initialize VLA cognitive planning system

        Args:
            robot_interface: Interface to the target robot
            api_key: LLM API key
        """
        self.robot_interface = robot_interface
        self.llm_generator = LLMPlanGenerator(api_key)
        self.plan_executor = PlanExecutor(robot_interface)
        self.plan_monitor = PlanMonitor(robot_interface, self.plan_executor)
        self.htn_planner = HTNPlanner(self.llm_generator)
        self.uncertainty_handler = UncertaintyHandler()

        # Current state tracking
        self.current_environment = EnvironmentState()
        self.current_robot_capabilities = RobotCapabilities()
        self.active_plan = None

    def process_high_level_command(self, command: str) -> bool:
        """
        Process a high-level natural language command through the full pipeline

        Args:
            command: Natural language command from user

        Returns:
            True if command was successfully processed and executed
        """
        print(f"Processing command: {command}")

        # Update environment state
        self._update_environment_state()

        # Generate plan using LLM
        plan = self.llm_generator.generate_plan(
            command,
            self.current_environment,
            self.current_robot_capabilities
        )

        if not plan:
            print("Failed to generate plan for command")
            return False

        # Validate plan
        is_valid, errors = plan.validate()
        if not is_valid:
            print(f"Generated plan has errors: {errors}")
            return False

        print(f"Generated plan with {len(plan.steps)} steps")

        # Execute plan with monitoring
        self.plan_monitor.start_monitoring()

        def status_callback(status):
            print(f"Plan execution status: {status.value}")

        success = self.plan_executor.execute_plan(plan, status_callback)

        self.plan_monitor.stop_monitoring()

        if success:
            print("Plan executed successfully")
        else:
            print("Plan execution failed")

        return success

    def _update_environment_state(self):
        """Update environment state from robot sensors"""
        # Get object information from perception
        objects = self.robot_interface.get_perceived_objects()
        for obj_id, obj_info in objects.items():
            self.current_environment.update_object(obj_id, obj_info)

        # Get location information
        robot_location = self.robot_interface.get_robot_location()
        self.current_environment.robot_state["position"] = robot_location

        # Get battery status
        battery_level = self.robot_interface.get_battery_level()
        self.current_environment.robot_state["battery"] = battery_level

        # Get other environmental constraints
        obstacles = self.robot_interface.get_detected_obstacles()
        if obstacles:
            self.current_environment.add_constraint(f"obstacles_detected: {len(obstacles)} obstacles")

    def handle_environment_change(self, changes: Dict):
        """Handle environment changes detected during execution"""
        print(f"Environment change detected: {changes}")

        # If there's an active plan and it's affected by changes, consider replanning
        if (self.active_plan and
            self.plan_executor.current_status == ExecutionStatus.EXECUTING):

            # Check if changes affect current plan
            if self._changes_affect_active_plan(changes):
                print("Environment changes affect active plan, considering replanning")
                self._handle_replanning(changes)

    def _changes_affect_active_plan(self, changes: Dict) -> bool:
        """Check if environment changes affect the currently executing plan"""
        # This would check the current plan against the changes
        # For now, return True for any significant changes
        return bool(changes)

    def _handle_replanning(self, changes: Dict):
        """Handle replanning when environment changes affect current plan"""
        # Pause current execution
        self.plan_executor.pause_execution()

        # Update environment state with changes
        self._update_environment_state()

        # Generate new plan considering changes
        new_plan = self.llm_generator.refine_plan(
            self.active_plan,
            f"Environment changed: {changes}. Adjust plan accordingly."
        )

        if new_plan:
            # Resume execution with new plan
            self.active_plan = new_plan
            self.plan_executor.resume_execution()
        else:
            # If replanning fails, cancel execution
            self.plan_executor.cancel_execution()

    def setup_monitoring_callbacks(self):
        """Setup callbacks for monitoring and adaptation"""
        self.plan_monitor.add_environment_change_callback(
            self.handle_environment_change
        )

    def continuous_operation_loop(self):
        """Run continuous operation loop waiting for commands"""
        print("Starting VLA cognitive planning system...")
        self.setup_monitoring_callbacks()

        try:
            while True:
                # In a real system, this would wait for commands from voice system
                # or other input sources
                command = self._get_next_command()
                if command:
                    self.process_high_level_command(command)

                time.sleep(0.1)  # Small delay to prevent busy waiting

        except KeyboardInterrupt:
            print("\nShutting down cognitive planning system...")
            self.plan_executor.cancel_execution()

    def _get_next_command(self) -> str:
        """Get the next command to process (placeholder implementation)"""
        # This would interface with the voice-to-action system
        # or other command sources
        return None  # Placeholder - would get actual commands in real implementation
```

## 3.7 Performance Considerations

### 3.7.1 Caching and Optimization

Optimizing LLM-based planning for performance:

```python
import hashlib
from typing import Optional
from functools import lru_cache

class PlanCache:
    """Cache for storing and retrieving previously generated plans"""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []  # For LRU eviction

    def get_plan(self, task_hash: str) -> Optional[Plan]:
        """Get plan from cache by hash"""
        if task_hash in self.cache:
            # Update access order for LRU
            if task_hash in self.access_order:
                self.access_order.remove(task_hash)
            self.access_order.append(task_hash)
            return self.cache[task_hash]
        return None

    def put_plan(self, task_hash: str, plan: Plan):
        """Put plan in cache"""
        # Check if we need to evict an old entry
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            if self.access_order:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]

        self.cache[task_hash] = plan
        self.access_order.append(task_hash)

    def generate_task_hash(self, task_description: str, environment_state: EnvironmentState) -> str:
        """Generate hash for task and environment combination"""
        state_str = environment_state.to_text_description()
        combined = f"{task_description}||{state_str}"
        return hashlib.md5(combined.encode()).hexdigest()

class OptimizedLLMPlanGenerator(LLMPlanGenerator):
    """LLM plan generator with caching and optimization"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(api_key, model)
        self.cache = PlanCache()

    def generate_plan(self, task_description: str, environment_state: EnvironmentState,
                     robot_capabilities: RobotCapabilities) -> Optional[Plan]:
        """Generate plan with caching"""
        # Generate hash for this task/environment combination
        task_hash = self.cache.generate_task_hash(task_description, environment_state)

        # Check cache first
        cached_plan = self.cache.get_plan(task_hash)
        if cached_plan:
            print("Retrieved plan from cache")
            return cached_plan

        # Generate new plan
        plan = super().generate_plan(task_description, environment_state, robot_capabilities)

        # Cache the plan if successful
        if plan:
            self.cache.put_plan(task_hash, plan)
            print("Cached newly generated plan")

        return plan
```

## Summary

Cognitive planning using LLMs represents a significant advancement in robotic autonomy, enabling robots to understand and execute complex natural language commands. This chapter covered the complete pipeline from prompt engineering and plan generation to execution monitoring and adaptation. Key components include structured environment representation, LLM integration, plan validation, execution frameworks, and uncertainty handling.

The implementation provides a robust foundation for cognitive planning that can adapt to environmental changes and handle uncertainty. Future systems will likely incorporate more sophisticated reasoning, better uncertainty quantification, and tighter integration with perception and control systems.

## Exercises

1. Implement a plan refinement system that learns from execution failures to improve future plans.

2. Design a multi-agent planning system where multiple robots coordinate their plans using LLMs.

3. Create a system that can handle temporal constraints and deadlines in LLM-generated plans.

4. Implement a plan explanation system that can describe the reasoning behind LLM-generated plans to users.

5. Design an active learning system that queries users for clarification when LLM plans are ambiguous.