# Chapter 5: Reinforcement Learning & Sim-to-Real Transfer

## Learning Objectives

After completing this chapter, you will be able to:
- Implement reinforcement learning algorithms for humanoid robotics
- Apply domain randomization techniques to bridge sim-to-real gap
- Design reward functions for humanoid locomotion and manipulation
- Implement imitation learning for humanoid behaviors
- Validate and transfer policies from simulation to real robots

## Introduction to Reinforcement Learning for Humanoid Robots

Reinforcement Learning (RL) has emerged as a powerful approach for learning complex humanoid behaviors that are difficult to engineer manually. Humanoid robots require sophisticated control policies to handle the complex dynamics of bipedal locomotion, balance maintenance, and interaction with human environments. RL provides a framework to learn these behaviors through trial and error in simulation, which can then be transferred to real robots.

### Why RL for Humanoid Robotics?

1. **Complex Control**: Humanoid robots have high-dimensional action spaces and complex dynamics
2. **Adaptability**: RL policies can adapt to different terrains and conditions
3. **Generalization**: Well-trained policies can generalize to unseen scenarios
4. **Efficiency**: Learn optimal behaviors without explicit programming
5. **Robustness**: Policies can learn to recover from disturbances

### Challenges in Humanoid RL

- **High-dimensional state-action spaces**: 20+ joints require complex control policies
- **Balance requirements**: Maintaining stability during locomotion
- **Safety constraints**: Avoiding falls and self-collision
- **Sample efficiency**: Real robot training requires many samples
- **Sim-to-real gap**: Differences between simulation and reality

## Reinforcement Learning Frameworks for Humanoid Robots

### Isaac Lab for RL Training

Isaac Lab provides a comprehensive framework for training humanoid robots using reinforcement learning:

```python
# Isaac Lab example for humanoid locomotion
import torch
import omni.isaac.orbit_tasks.locomotion.velocity.mdp as mdp
from omni.isaac.orbit.assets import AssetBase
from omni.isaac.orbit.envs import RLTask
from omni.isaac.orbit.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.scene import InteractiveScene
from omni.isaac.orbit.utils import configclass

@configclass
class HumanoidEnvCfg:
    # Scene settings
    scene: InteractiveScene = None

    # Robot settings
    robot: AssetBase = SceneEntityCfg("robot", init_state=None)

    # Curriculum settings
    curriculum = {
        "spawn_position": CurrTerm(func=mdp.uniform_lin_vel_with_facing_direction),
    }

    # Termination settings
    terminations = {
        "time_out": DoneTerm(func=mdp.time_out),
        "base_height": DoneTerm(func=mdp.base_height, params={"threshold": 0.3}),
        "roll_pitch": DoneTerm(func=mdp.roll_pitch_at_limit),
    }

    # Event settings
    events = {
        "reset_robot_joints": mdp.JointPositionCommandNoiseCfg(
            asset_cfg=SceneEntityCfg("robot", joint_names=[".*"]),
            position_range=(0.0, 0.0),
            operation="add",
        ),
    }

class HumanoidLocomotionTask(RLTask):
    def __init__(self, cfg: HumanoidEnvCfg, sim_cfg):
        super().__init__(cfg, sim_cfg)

        # Initialize humanoid-specific components
        self._setup_humanoid_agents()
        self._setup_locomotion_controller()

    def _setup_humanoid_agents(self):
        """Setup humanoid robot agents"""
        # Implementation for humanoid-specific setup
        pass

    def _setup_locomotion_controller(self):
        """Setup locomotion controller"""
        # Implementation for walking controller
        pass

    def set_episode_length(self, env_ids):
        """Set episode length for humanoid locomotion"""
        # Custom episode termination for humanoid tasks
        pass

# Training configuration example
from omni.isaac.orbit_tasks.utils import train_cfg_parser
from omni.isaac.orbit_tasks.locomotion.velocity.config.unitree_a1 import agents

def train_humanoid_policy():
    """Train a humanoid locomotion policy using Isaac Lab"""
    # Parse training configuration
    env_cfg, agent_cfg = train_cfg_parser(HumanoidEnvCfg, agents.CLIArgs())

    # Initialize RL training environment
    from omni.isaac.orbit_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg

    runner_cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=1500,
        save_interval=50,
        experiment_name="humanoid_locomotion",
        run_name="",
        logger="tensorboard",
        enable_wandb=False,
    )

    # Train the policy
    from omni.isaac.orbit_tasks.utils.train_utils.train import train
    ppo_runner = train(
        cfg=agent_cfg,
        env_cfg=env_cfg,
        init_helper_class_name="init_helper",
        logger_cfg=runner_cfg,
    )

    return ppo_runner
```

### Popular RL Algorithms for Humanoid Control

Different RL algorithms are suited for different humanoid control tasks:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HumanoidActorCritic(nn.Module):
    """Actor-Critic network for humanoid control"""

    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions are bounded for humanoid joints
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        action_mean = self.actor(state)
        state_value = self.critic(state)
        return action_mean, state_value

class PPOTrainer:
    """Proximal Policy Optimization for humanoid learning"""

    def __init__(self, state_dim, action_dim, lr=3e-4, clip_epsilon=0.2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor_critic = HumanoidActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)

        self.clip_epsilon = clip_epsilon
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.5

    def compute_action(self, state, deterministic=False):
        """Compute action and log probability"""
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action_mean, state_value = self.actor_critic(state)

        if deterministic:
            action = action_mean
        else:
            # Add noise for exploration
            action = action_mean + torch.randn_like(action_mean) * 0.1

        # Clamp actions to humanoid joint limits
        action = torch.clamp(action, -1.0, 1.0)

        return action.detach().cpu().numpy()[0], state_value.detach().cpu().numpy()[0]

    def update(self, states, actions, old_log_probs, returns, advantages):
        """Update policy using PPO objective"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Current policy evaluation
        new_action_means, state_values = self.actor_critic(states)

        # Calculate new log probabilities
        # For simplicity, using a Gaussian policy with fixed variance
        dist = torch.distributions.Normal(new_action_means, 0.5)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)

        # Calculate ratio
        ratios = torch.exp(new_log_probs - old_log_probs)

        # PPO objective
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(state_values.squeeze(), returns)

        # Entropy loss for exploration
        entropy = dist.entropy().mean()

        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        # Update network
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        self.optimizer.step()

        return total_loss.item(), policy_loss.item(), value_loss.item()
```

## Sim-to-Real Transfer Techniques

### Domain Randomization

Domain randomization is a key technique to bridge the sim-to-real gap by randomizing simulation parameters:

```python
import numpy as np
import random

class DomainRandomization:
    """Domain randomization for sim-to-real transfer"""

    def __init__(self):
        # Physics parameters randomization ranges
        self.randomization_params = {
            # Robot properties
            'mass_multiplier_range': (0.8, 1.2),
            'friction_range': (0.4, 1.6),
            'restitution_range': (0.0, 0.2),

            # Control parameters
            'pd_gains_range': (0.8, 1.2),
            'actuator_delay_range': (0.0, 0.02),  # 0-20ms delay

            # Sensor noise
            'imu_noise_std': (0.001, 0.01),
            'joint_pos_noise_std': (0.001, 0.01),
            'joint_vel_noise_std': (0.01, 0.1),

            # Environment parameters
            'ground_friction_range': (0.4, 1.6),
            'lighting_condition_range': (0.5, 2.0),
            'texture_randomization': True,

            # External disturbances
            'push_force_range': (0.0, 50.0),  # Newtons
            'push_interval_range': (100, 500)  # steps between pushes
        }

    def randomize_robot_properties(self, robot):
        """Randomize robot physical properties"""
        # Randomize link masses
        for link_idx in range(robot.num_links):
            original_mass = robot.get_link_mass(link_idx)
            mass_multiplier = np.random.uniform(
                self.randomization_params['mass_multiplier_range'][0],
                self.randomization_params['mass_multiplier_range'][1]
            )
            new_mass = original_mass * mass_multiplier
            robot.set_link_mass(link_idx, new_mass)

        # Randomize friction coefficients
        for joint_idx in range(robot.num_joints):
            friction_range = self.randomization_params['friction_range']
            friction = np.random.uniform(friction_range[0], friction_range[1])
            robot.set_joint_friction(joint_idx, friction)

    def randomize_control_parameters(self):
        """Randomize control parameters"""
        gains_range = self.randomization_params['pd_gains_range']
        kp_randomization = np.random.uniform(gains_range[0], gains_range[1])
        kd_randomization = np.random.uniform(gains_range[0], gains_range[1])

        return {
            'kp_scale': kp_randomization,
            'kd_scale': kd_randomization,
            'delay': np.random.uniform(
                self.randomization_params['actuator_delay_range'][0],
                self.randomization_params['actuator_delay_range'][1]
            )
        }

    def randomize_sensors(self, observation):
        """Add random noise to sensor observations"""
        # Add IMU noise
        imu_noise_range = self.randomization_params['imu_noise_std']
        imu_noise_std = np.random.uniform(imu_noise_range[0], imu_noise_range[1])
        observation['imu'] += np.random.normal(0, imu_noise_std, observation['imu'].shape)

        # Add joint position noise
        joint_pos_noise_range = self.randomization_params['joint_pos_noise_std']
        joint_pos_noise_std = np.random.uniform(joint_pos_noise_range[0], joint_pos_noise_range[1])
        observation['joint_pos'] += np.random.normal(0, joint_pos_noise_std, observation['joint_pos'].shape)

        # Add joint velocity noise
        joint_vel_noise_range = self.randomization_params['joint_vel_noise_std']
        joint_vel_noise_std = np.random.uniform(joint_vel_noise_range[0], joint_vel_noise_range[1])
        observation['joint_vel'] += np.random.normal(0, joint_vel_noise_std, observation['joint_vel'].shape)

        return observation

    def apply_external_disturbances(self, step_count):
        """Apply random external disturbances"""
        if step_count % np.random.randint(
            self.randomization_params['push_interval_range'][0],
            self.randomization_params['push_interval_range'][1]
        ) == 0:

            push_force_range = self.randomization_params['push_force_range']
            push_force = np.random.uniform(push_force_range[0], push_force_range[1])

            # Apply force in random direction
            angle = np.random.uniform(0, 2 * np.pi)
            force_x = push_force * np.cos(angle)
            force_y = push_force * np.sin(angle)

            return [force_x, force_y, 0.0]  # [force_x, force_y, force_z]

        return [0.0, 0.0, 0.0]  # No disturbance

class HumanoidEnvironmentWithDR:
    """Humanoid environment with domain randomization"""

    def __init__(self):
        self.domain_randomizer = DomainRandomization()
        self.episode_step_count = 0

    def reset(self):
        """Reset environment with domain randomization"""
        # Randomize robot properties at episode start
        self.domain_randomizer.randomize_robot_properties(self.robot)

        # Randomize control parameters
        self.control_params = self.domain_randomizer.randomize_control_parameters()

        # Reset step counter
        self.episode_step_count = 0

        # Reset robot position and get initial observation
        obs = self.get_observation()
        return self.domain_randomizer.randomize_sensors(obs)

    def step(self, action):
        """Step environment with domain randomization"""
        self.episode_step_count += 1

        # Apply action with randomized control parameters
        randomized_action = self.apply_control_randomization(action)

        # Step the physics simulation
        self.apply_action(randomized_action)
        self.step_physics()

        # Apply external disturbances
        external_force = self.domain_randomizer.apply_external_disturbances(self.episode_step_count)
        self.apply_external_force(external_force)

        # Get observation with sensor noise
        obs = self.get_observation()
        obs = self.domain_randomizer.randomize_sensors(obs)

        # Calculate reward and check termination
        reward = self.calculate_reward()
        terminated = self.check_termination()
        truncated = self.episode_step_count >= self.max_episode_steps

        return obs, reward, terminated, truncated, {}

    def apply_control_randomization(self, action):
        """Apply control parameter randomization to action"""
        # Scale PD gains
        kp_scaled = self.control_params['kp_scale']
        kd_scaled = self.control_params['kd_scale']

        # Apply delay simulation
        delay_steps = int(self.control_params['delay'] / self.sim_dt)

        # In a real implementation, you'd store previous actions and apply delay
        # For simplicity, just return the action
        return action * kp_scaled  # Simplified for example

    def get_observation(self):
        """Get current observation from robot sensors"""
        obs = {
            'joint_pos': self.robot.get_joint_positions(),
            'joint_vel': self.robot.get_joint_velocities(),
            'imu': self.robot.get_imu_data(),
            'base_pos': self.robot.get_base_position(),
            'base_quat': self.robot.get_base_orientation(),
        }
        return obs

    def calculate_reward(self):
        """Calculate reward for humanoid locomotion"""
        # Get current robot state
        base_pos = self.robot.get_base_position()
        base_vel = self.robot.get_base_velocity()
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()

        # Reward for forward velocity
        forward_vel_reward = base_vel[0] * 2.0  # Encourage forward movement

        # Penalty for deviation from upright position
        base_quat = self.robot.get_base_orientation()
        z_axis = np.array([0, 0, 1])
        world_up = np.array([0, 0, 1])  # In base's frame
        upright_penalty = -abs(np.dot(base_quat[:3], world_up) - 1.0) * 10.0

        # Penalty for joint limits
        joint_limit_penalty = 0.0
        for pos in joint_pos:
            if abs(pos) > 2.0:  # Example joint limit
                joint_limit_penalty -= 1.0

        # Penalty for excessive joint velocity
        velocity_penalty = -np.sum(np.abs(joint_vel)) * 0.01

        # Reward for stability (not falling)
        height_reward = max(0, self.robot.get_base_position()[2] - 0.3) * 5.0

        total_reward = (
            forward_vel_reward +
            upright_penalty +
            joint_limit_penalty +
            velocity_penalty +
            height_reward
        )

        return max(total_reward, -10.0)  # Clamp reward

    def check_termination(self):
        """Check if episode should terminate"""
        base_pos = self.robot.get_base_position()
        base_quat = self.robot.get_base_orientation()

        # Terminate if robot falls (too low or too tilted)
        fallen = base_pos[2] < 0.2  # Too low
        tilted = abs(base_quat[2]) < 0.5  # Not upright enough (simplified check)

        return fallen or tilted
```

### Domain Adaptation Techniques

Domain adaptation helps improve sim-to-real transfer by adapting policies to real-world characteristics:

```python
import torch
import torch.nn as nn

class DomainAdaptationNetwork(nn.Module):
    """Network for domain adaptation"""

    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()

        # Feature extractor shared between domains
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Domain classifier to distinguish sim vs real
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Probability of being real data
        )

        # Task-specific predictor
        self.task_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Example: for value prediction
        )

    def forward(self, x, domain_label=None):
        features = self.feature_extractor(x)

        # Task prediction
        task_output = self.task_predictor(features)

        # Domain prediction (only when training)
        domain_output = None
        if domain_label is not None:
            domain_output = self.domain_classifier(features)

        return task_output, domain_output

class DomainAdversarialTrainer:
    """Domain adversarial training for sim-to-real transfer"""

    def __init__(self, state_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.da_network = DomainAdaptationNetwork(state_dim).to(self.device)
        self.task_optimizer = torch.optim.Adam(
            list(self.da_network.feature_extractor.parameters()) +
            list(self.da_network.task_predictor.parameters()),
            lr=1e-4
        )
        self.domain_optimizer = torch.optim.Adam(
            list(self.da_network.feature_extractor.parameters()) +
            list(self.da_network.domain_classifier.parameters()),
            lr=1e-4
        )

        self.domain_criterion = nn.BCELoss()
        self.task_criterion = nn.MSELoss()

    def train_step(self, sim_states, real_states, sim_targets, real_targets):
        """Train with domain adversarial loss"""
        # Convert to tensors
        sim_states = torch.FloatTensor(sim_states).to(self.device)
        real_states = torch.FloatTensor(real_states).to(self.device)
        sim_targets = torch.FloatTensor(sim_targets).to(self.device)
        real_targets = torch.FloatTensor(real_targets).to(self.device)

        # Create domain labels (0 for sim, 1 for real)
        sim_domains = torch.zeros(sim_states.size(0), 1).to(self.device)
        real_domains = torch.ones(real_states.size(0), 1).to(self.device)

        # Train domain classifier to distinguish domains (maximize loss)
        self.domain_optimizer.zero_grad()

        _, sim_domain_pred = self.da_network(sim_states, domain_label=True)
        _, real_domain_pred = self.da_network(real_states, domain_label=True)

        domain_loss = (
            self.domain_criterion(sim_domain_pred, sim_domains) +
            self.domain_criterion(real_domain_pred, real_domains)
        )

        domain_loss.backward()
        self.domain_optimizer.step()

        # Train feature extractor to fool domain classifier (minimize loss)
        # and task predictor to minimize task loss
        self.task_optimizer.zero_grad()

        sim_task_pred, sim_domain_pred = self.da_network(sim_states, domain_label=True)
        real_task_pred, real_domain_pred = self.da_network(real_states, domain_label=True)

        # Task loss
        task_loss = (
            self.task_criterion(sim_task_pred, sim_targets) +
            self.task_criterion(real_task_pred, real_targets)
        )

        # Domain confusion loss (minimize domain prediction accuracy)
        domain_confusion_loss = (
            self.domain_criterion(sim_domain_pred, 1 - sim_domains) +  # Try to predict 1
            self.domain_criterion(real_domain_pred, 1 - real_domains)  # Try to predict 0
        )

        total_loss = task_loss + domain_confusion_loss
        total_loss.backward()
        self.task_optimizer.step()

        return task_loss.item(), domain_loss.item()
```

## Imitation Learning for Humanoid Robots

### Behavior Cloning

Behavior cloning learns from expert demonstrations:

```python
import torch
import torch.nn as nn
import numpy as np

class BehaviorCloningPolicy(nn.Module):
    """Simple behavior cloning policy"""

    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions bounded to [-1, 1]
        )

    def forward(self, state):
        return self.network(state)

class BehaviorCloningTrainer:
    """Behavior cloning trainer"""

    def __init__(self, state_dim, action_dim, lr=1e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = BehaviorCloningPolicy(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_epoch(self, expert_states, expert_actions, batch_size=64):
        """Train for one epoch on expert data"""
        dataset_size = len(expert_states)

        # Shuffle data
        indices = np.random.permutation(dataset_size)
        expert_states = expert_states[indices]
        expert_actions = expert_actions[indices]

        total_loss = 0
        num_batches = 0

        for i in range(0, dataset_size, batch_size):
            batch_states = expert_states[i:i+batch_size]
            batch_actions = expert_actions[i:i+batch_size]

            # Convert to tensors
            states_tensor = torch.FloatTensor(batch_states).to(self.device)
            actions_tensor = torch.FloatTensor(batch_actions).to(self.device)

            # Forward pass
            predicted_actions = self.policy(states_tensor)

            # Compute loss
            loss = self.criterion(predicted_actions, actions_tensor)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

# Example usage for collecting expert demonstrations
class ExpertDemonstrationCollector:
    """Collect expert demonstrations for behavior cloning"""

    def __init__(self, robot_interface):
        self.robot = robot_interface
        self.demonstrations = {
            'states': [],
            'actions': [],
            'rewards': [],
            'next_states': []
        }

    def collect_demonstration(self, expert_policy, num_episodes=10):
        """Collect demonstrations using expert policy"""
        for episode in range(num_episodes):
            obs = self.robot.reset()
            episode_states = []
            episode_actions = []

            for step in range(1000):  # Example episode length
                # Get expert action
                action = expert_policy(obs)

                # Store state and action
                episode_states.append(obs.copy())
                episode_actions.append(action.copy())

                # Apply action
                next_obs, reward, done, info = self.robot.step(action)

                if done:
                    break

                obs = next_obs

            # Store episode data
            self.demonstrations['states'].extend(episode_states)
            self.demonstrations['actions'].extend(episode_actions)

    def get_training_data(self):
        """Get collected demonstrations as numpy arrays"""
        states = np.array(self.demonstrations['states'])
        actions = np.array(self.demonstrations['actions'])

        return states, actions
```

### Generative Adversarial Imitation Learning (GAIL)

GAIL learns policies that are indistinguishable from expert demonstrations:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GAILDiscriminator(nn.Module):
    """Discriminator for GAIL - distinguishes expert vs policy data"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Probability of being expert data
        )

    def forward(self, state, action):
        inputs = torch.cat([state, action], dim=-1)
        return self.network(inputs)

class GAILTrainer:
    """GAIL trainer for humanoid imitation learning"""

    def __init__(self, state_dim, action_dim, policy_lr=3e-4, disc_lr=3e-4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Policy network (actor)
        self.policy = BehaviorCloningPolicy(state_dim, action_dim).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)

        # Discriminator network
        self.discriminator = GAILDiscriminator(state_dim, action_dim).to(self.device)
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=disc_lr)

        self.bce_loss = nn.BCELoss()

    def compute_reward(self, state, action):
        """Compute reward as -log(D(s,a)) for policy optimization"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            action_tensor = torch.FloatTensor(action).to(self.device)

            disc_output = self.discriminator(state_tensor, action_tensor)
            # Use log(D) as reward (when D is probability of expert)
            reward = -torch.log(disc_output + 1e-8)  # Add small value to avoid log(0)

        return reward.cpu().numpy()

    def discriminator_loss(self, expert_states, expert_actions, policy_states, policy_actions):
        """Compute discriminator loss to distinguish expert vs policy data"""
        # Convert to tensors
        expert_states = torch.FloatTensor(expert_states).to(self.device)
        expert_actions = torch.FloatTensor(expert_actions).to(self.device)
        policy_states = torch.FloatTensor(policy_states).to(self.device)
        policy_actions = torch.FloatTensor(policy_actions).to(self.device)

        # Labels: 1 for expert, 0 for policy
        expert_labels = torch.ones(expert_states.size(0), 1).to(self.device)
        policy_labels = torch.zeros(policy_states.size(0), 1).to(self.device)

        # Forward pass
        expert_logits = self.discriminator(expert_states, expert_actions)
        policy_logits = self.discriminator(policy_states, policy_actions)

        # Discriminator loss - maximize probability of correct classification
        disc_loss = (
            self.bce_loss(expert_logits, expert_labels) +
            self.bce_loss(policy_logits, policy_labels)
        )

        return disc_loss

    def update_discriminator(self, expert_buffer, policy_buffer, num_updates=5):
        """Update discriminator to better distinguish expert vs policy data"""
        for _ in range(num_updates):
            # Sample from both buffers
            exp_batch = expert_buffer.sample(64)
            pol_batch = policy_buffer.sample(64)

            self.disc_optimizer.zero_grad()

            disc_loss = self.discriminator_loss(
                exp_batch['states'], exp_batch['actions'],
                pol_batch['states'], pol_batch['actions']
            )

            disc_loss.backward()
            self.disc_optimizer.step()

    def update_policy(self, states, actions, advantages):
        """Update policy using computed rewards"""
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Compute log probabilities under current policy
        predicted_actions = self.policy(states)

        # Use advantage weighted loss
        policy_loss = -(advantages * torch.mean((predicted_actions - actions)**2, dim=1)).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return policy_loss.item()
```

## Reward Engineering for Humanoid Tasks

### Locomotion Rewards

Designing effective reward functions is crucial for learning successful humanoid behaviors:

```python
import numpy as np

class HumanoidLocomotionRewards:
    """Reward functions for humanoid locomotion tasks"""

    def __init__(self, target_velocity=1.0, target_direction=np.array([1.0, 0.0, 0.0])):
        self.target_velocity = target_velocity
        self.target_direction = target_direction / np.linalg.norm(target_direction)

        # Reward weights
        self.weights = {
            'velocity': 1.0,
            'upright': 1.0,
            'energy': 0.05,
            'joint_regularization': 0.1,
            'action_smoothness': 0.01,
            'survival': 0.1,
            'foot_placement': 0.5
        }

    def compute_reward(self, robot_state, action, dt):
        """Compute comprehensive reward for locomotion"""
        reward = 0.0

        # Velocity reward - encourage forward motion at target speed
        current_velocity = self._get_base_velocity(robot_state)
        forward_vel = np.dot(current_velocity[:2], self.target_direction[:2])
        velocity_reward = self.weights['velocity'] * (
            forward_vel - abs(forward_vel - self.target_velocity) * 0.5
        )
        reward += velocity_reward

        # Upright posture reward - maintain balance
        upright_reward = self.weights['upright'] * self._compute_upright_reward(robot_state)
        reward += upright_reward

        # Energy efficiency reward - minimize joint actuation
        energy_reward = -self.weights['energy'] * np.sum(np.abs(action))
        reward += energy_reward

        # Joint regularization - keep joints near default positions
        joint_reg_reward = self.weights['joint_regularization'] * self._compute_joint_regularization(robot_state)
        reward += joint_reg_reward

        # Action smoothness - penalize jerky movements
        action_smoothness = self.weights['action_smoothness'] * self._compute_action_smoothness(action, dt)
        reward += action_smoothness

        # Survival reward - stay alive and not fallen
        survival_reward = self.weights['survival'] * self._compute_survival_reward(robot_state)
        reward += survival_reward

        # Foot placement reward - encourage proper stepping pattern
        foot_placement_reward = self.weights['foot_placement'] * self._compute_foot_placement_reward(robot_state)
        reward += foot_placement_reward

        return reward

    def _get_base_velocity(self, robot_state):
        """Extract base velocity from robot state"""
        # This would interface with your robot's velocity estimation
        return robot_state['base_vel']  # Example: [vx, vy, vz]

    def _compute_upright_reward(self, robot_state):
        """Reward for maintaining upright posture"""
        base_quat = robot_state['base_quat']  # [x, y, z, w]

        # Convert quaternion to rotation matrix to get world up vector
        qw, qx, qy, qz = base_quat
        # Get the world Z-axis (up) vector in robot's frame
        world_up = np.array([
            2 * (qx*qz + qw*qy),
            2 * (qy*qz - qw*qx),
            1 - 2 * (qx*qx + qy*qy)
        ])

        # Reward alignment with world up (z-axis)
        z_alignment = world_up[2]  # Dot product with [0, 0, 1]
        return max(0, z_alignment)  # Only positive reward for upright

    def _compute_joint_regularization(self, robot_state):
        """Reward for keeping joints near default positions"""
        joint_pos = robot_state['joint_pos']
        default_pos = robot_state['default_joint_pos']  # Predefined default positions

        # Penalize deviation from default positions
        deviation = np.sum((joint_pos - default_pos)**2)
        return np.exp(-deviation)  # Gaussian-like reward

    def _compute_action_smoothness(self, action, dt):
        """Reward for smooth actions"""
        # This would typically use previous actions
        # For simplicity, we'll reward smaller action magnitudes
        return np.exp(-np.sum(action**2))

    def _compute_survival_reward(self, robot_state):
        """Reward for staying alive (not fallen)"""
        base_pos = robot_state['base_pos']
        base_quat = robot_state['base_quat']

        # Check if fallen based on height and orientation
        height_ok = base_pos[2] > 0.3  # Minimum height
        upright_ok = self._is_upright(base_quat)  # Check orientation

        return 1.0 if (height_ok and upright_ok) else -1.0

    def _is_upright(self, quat):
        """Check if robot is in upright position"""
        qw, qx, qy, qz = quat
        # Simplified check - world z-axis should be mostly aligned
        world_up_z = 1 - 2 * (qx*qx + qy*qy)
        return world_up_z > 0.5  # Reasonably upright

    def _compute_foot_placement_reward(self, robot_state):
        """Reward for proper foot placement during walking"""
        # This would analyze foot positions and contact patterns
        # For simplicity, return a basic reward based on stance
        left_foot_pos = robot_state.get('left_foot_pos', [0, 0, 0])
        right_foot_pos = robot_state.get('right_foot_pos', [0, 0, 0])
        base_pos = robot_state['base_pos']

        # Encourage feet to be appropriately positioned relative to body
        left_dist = abs(left_foot_pos[1] - base_pos[1])  # Lateral distance
        right_dist = abs(right_foot_pos[1] - base_pos[1])  # Lateral distance

        # Reward for appropriate stance width
        stance_width = abs(left_foot_pos[1] - right_foot_pos[1])
        optimal_stance = 0.2  # 20cm apart
        stance_reward = np.exp(-abs(stance_width - optimal_stance))

        return stance_reward

class ManipulationRewards:
    """Reward functions for humanoid manipulation tasks"""

    def __init__(self):
        self.weights = {
            'reach_object': 1.0,
            'grasp_object': 5.0,
            'lift_object': 2.0,
            'transport_object': 3.0,
            'avoid_collisions': 1.0,
            'energy_efficiency': 0.1
        }

    def compute_manipulation_reward(self, robot_state, task_params):
        """Compute reward for manipulation task"""
        reward = 0.0

        # Get relevant states
        ee_pos = robot_state['end_effector_pos']
        object_pos = task_params['target_object_pos']
        grasp_status = robot_state['grasp_status']
        object_height = robot_state['object_height']

        # Reach object reward
        dist_to_object = np.linalg.norm(ee_pos - object_pos)
        reach_reward = self.weights['reach_object'] * np.exp(-dist_to_object)
        reward += reach_reward

        # Grasp reward (if object is grasped)
        if grasp_status:
            reward += self.weights['grasp_object']

            # Lift object reward (if object is lifted)
            if object_height > 0.1:  # Object is lifted above ground
                reward += self.weights['lift_object']

                # Transport reward (if moving object to target)
                target_pos = task_params['target_pos']
                obj_to_target_dist = np.linalg.norm(object_pos - target_pos)
                transport_reward = self.weights['transport_object'] * np.exp(-obj_to_target_dist)
                reward += transport_reward

        # Collision avoidance penalty
        collision_penalty = -self.weights['avoid_collisions'] * robot_state['collision_count']
        reward += collision_penalty

        # Energy efficiency
        energy_penalty = -self.weights['energy_efficiency'] * np.sum(np.abs(robot_state['joint_vel']))
        reward += energy_penalty

        return reward
```

## Policy Transfer and Validation

### Safety Considerations for Real Robot Deployment

```python
import numpy as np

class PolicySafetyValidator:
    """Validate RL policies before real robot deployment"""

    def __init__(self, robot_specifications):
        self.max_joint_velocities = robot_specifications['max_joint_velocities']
        self.max_joint_torques = robot_specifications['max_joint_torques']
        self.joint_limits = robot_specifications['joint_limits']
        self.safety_margin = 0.1  # 10% safety margin

    def validate_action(self, action, current_state):
        """Validate that action is safe for real robot"""
        safety_report = {
            'is_safe': True,
            'violations': [],
            'suggested_action': action.copy()
        }

        # Check joint velocity limits
        desired_velocities = action  # Assuming action represents desired velocities
        for i, (vel, max_vel) in enumerate(zip(desired_velocities, self.max_joint_velocities)):
            if abs(vel) > max_vel * (1 - self.safety_margin):
                safety_report['is_safe'] = False
                safety_report['violations'].append(
                    f"Joint {i} velocity {abs(vel):.3f} exceeds limit {max_vel * (1 - self.safety_margin):.3f}"
                )

                # Suggest safer velocity
                safety_report['suggested_action'][i] = np.clip(
                    vel,
                    -max_vel * (1 - self.safety_margin),
                    max_vel * (1 - self.safety_margin)
                )

        # Check joint position limits
        current_positions = current_state['joint_pos']
        suggested_positions = current_positions + desired_velocities * 0.01  # Assuming 10ms control cycle

        for i, (pos, (min_pos, max_pos)) in enumerate(zip(suggested_positions, self.joint_limits)):
            if pos < min_pos * (1 + self.safety_margin) or pos > max_pos * (1 - self.safety_margin):
                safety_report['is_safe'] = False
                safety_report['violations'].append(
                    f"Joint {i} position {pos:.3f} would exceed limits [{min_pos:.3f}, {max_pos:.3f}]"
                )

                # Suggest position within limits
                safety_report['suggested_action'][i] = np.clip(
                    pos,
                    min_pos * (1 + self.safety_margin),
                    max_pos * (1 - self.safety_margin)
                ) - current_positions[i]  # Convert back to velocity

        # Check for potential self-collisions
        # This would require more complex geometric checking
        # For now, we'll assume a simple check
        if self._check_self_collision(current_positions, suggested_positions):
            safety_report['is_safe'] = False
            safety_report['violations'].append("Action may cause self-collision")

        return safety_report

    def _check_self_collision(self, current_pos, target_pos):
        """Simple check for potential self-collisions"""
        # This is a simplified collision check
        # In reality, you'd need geometric collision detection
        for i in range(len(current_pos)):
            if abs(target_pos[i] - current_pos[i]) > 0.5:  # Large movement threshold
                # Check if this joint movement could cause collision
                # This is a placeholder - implement proper collision checking
                pass
        return False

class TransferLearningManager:
    """Manage the transfer of policies from sim to real"""

    def __init__(self):
        self.sim_to_real_calibration = {}
        self.performance_threshold = 0.7  # Minimum performance in sim for transfer

    def calibrate_sensor_mapping(self, sim_sensor_data, real_sensor_data):
        """Calibrate mapping between sim and real sensor data"""
        # Create calibration mapping
        self.sim_to_real_calibration['imu'] = self._calibrate_imu(sim_sensor_data, real_sensor_data)
        self.sim_to_real_calibration['joint_pos'] = self._calibrate_joint_pos(sim_sensor_data, real_sensor_data)
        self.sim_to_real_calibration['force_torque'] = self._calibrate_force_torque(sim_sensor_data, real_sensor_data)

    def _calibrate_imu(self, sim_data, real_data):
        """Calibrate IMU data between sim and real"""
        # Calculate scaling and bias factors
        sim_acc_mean = np.mean(sim_data['acc'], axis=0)
        real_acc_mean = np.mean(real_data['acc'], axis=0)

        sim_acc_std = np.std(sim_data['acc'], axis=0)
        real_acc_std = np.std(real_data['acc'], axis=0)

        # Calculate transformation parameters
        scale = real_acc_std / (sim_acc_std + 1e-8)  # Add small value to avoid division by zero
        bias = real_acc_mean - sim_acc_mean * scale

        return {'scale': scale, 'bias': bias}

    def _calibrate_joint_pos(self, sim_data, real_data):
        """Calibrate joint position data"""
        # For joint positions, we often need offset calibration
        offset = np.mean(real_data['joint_pos'] - sim_data['joint_pos'], axis=0)
        return {'offset': offset}

    def _calibrate_force_torque(self, sim_data, real_data):
        """Calibrate force/torque sensor data"""
        # Calculate scaling factors for force/torque sensors
        sim_force_norm = np.linalg.norm(sim_data['force'], axis=1)
        real_force_norm = np.linalg.norm(real_data['force'], axis=1)

        scale_factor = np.mean(real_force_norm / (sim_force_norm + 1e-8))
        return {'scale': scale_factor}

    def apply_calibration(self, sim_observation):
        """Apply calibration to sim observation for real robot"""
        calibrated_obs = sim_observation.copy()

        # Apply IMU calibration
        if 'imu' in calibrated_obs and 'imu' in self.sim_to_real_calibration:
            cal = self.sim_to_real_calibration['imu']
            calibrated_obs['imu'] = calibrated_obs['imu'] * cal['scale'] + cal['bias']

        # Apply joint position calibration
        if 'joint_pos' in calibrated_obs and 'joint_pos' in self.sim_to_real_calibration:
            cal = self.sim_to_real_calibration['joint_pos']
            calibrated_obs['joint_pos'] = calibrated_obs['joint_pos'] + cal['offset']

        return calibrated_obs

    def gradual_transfer_protocol(self, policy, validation_env, transfer_steps=100):
        """Gradually transfer policy from sim to real with validation"""
        transfer_log = {
            'steps': [],
            'performance': [],
            'safety_incidents': 0,
            'transfer_successful': False
        }

        for step in range(transfer_steps):
            # Start with mostly simulation, gradually increase real robot usage
            sim_ratio = max(0.1, 1.0 - (step / transfer_steps) * 0.9)

            if np.random.random() < sim_ratio:
                # Use simulation for training
                obs, reward, done, info = validation_env.step_in_simulation()
            else:
                # Use real robot for validation (safely)
                obs, reward, done, info = validation_env.step_on_real_robot()

            # Log performance
            transfer_log['steps'].append(step)
            transfer_log['performance'].append(reward)

            if info.get('safety_violation', False):
                transfer_log['safety_incidents'] += 1

            # Check if performance is adequate for full transfer
            recent_performance = np.mean(transfer_log['performance'][-10:])
            if recent_performance >= self.performance_threshold and step > 50:
                transfer_log['transfer_successful'] = True
                break

        return transfer_log
```

## Best Practices for RL in Humanoid Robotics

### 1. Safety-First Approach
- Implement comprehensive safety checks before real robot deployment
- Use simulation extensively for initial training
- Gradual transfer protocols with human oversight
- Emergency stop mechanisms

### 2. Sample Efficiency
- Use domain randomization to maximize simulation utility
- Implement curriculum learning
- Use prior knowledge and demonstrations
- Leverage transfer learning between tasks

### 3. Reward Design
- Design reward functions that promote stable behaviors
- Avoid reward hacking through careful shaping
- Include regularization terms for smooth movements
- Balance multiple objectives appropriately

### 4. Validation and Testing
- Extensive simulation validation before real robot testing
- Systematic safety checks at every stage
- Performance monitoring during deployment
- Continuous learning and adaptation

## Summary

This chapter covered reinforcement learning and sim-to-real transfer techniques for humanoid robots, including domain randomization, imitation learning, reward engineering, and safety considerations. RL provides powerful tools for learning complex humanoid behaviors, but requires careful consideration of safety, sample efficiency, and the sim-to-real gap. Successful deployment involves gradual transfer protocols and comprehensive validation.

## Exercises

1. Implement a PPO-based locomotion policy for a simulated humanoid
2. Apply domain randomization to improve sim-to-real transfer
3. Design reward functions for specific humanoid tasks
4. Implement a safety validation system for RL policies
5. Create an imitation learning system using expert demonstrations

## Further Reading

- "Reinforcement Learning: An Introduction" by Sutton & Barto
- "Deep Reinforcement Learning for Robotic Manipulation" by Gu et al.
- NVIDIA Isaac Lab Documentation
- "Learning Agile and Dynamic Motor Skills for Legged Robots"

---

*Next: [Module 4 Preface](../preface.md)*