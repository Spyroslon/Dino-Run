import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import DinoGame
import time

class DinoEnv(gym.Env):
    def __init__(self, logger=None):
        super().__init__()
        self.game = DinoGame()

        # Action space: 0=run, 1=jump, 2=duck, 3=fall, 4=stand
        self.action_space = spaces.Discrete(5)

        # Observation space
        self.max_obstacles = 3
        self.observation_space = spaces.Dict({
            "status": spaces.Discrete(5),
            "distance": spaces.Box(low=0, high=1e6, shape=(1,), dtype=np.float32),
            "speed": spaces.Box(low=0, high=100.0, shape=(1,), dtype=np.float32),
            "jump_velocity": spaces.Box(low=-50.0, high=50.0, shape=(1,), dtype=np.float32),
            "y_position": spaces.Box(low=0, high=200.0, shape=(1,), dtype=np.float32),
            "obstacles": spaces.Box(low=-100, high=1000.0, shape=(self.max_obstacles * 4,), dtype=np.float32)
        })
        
        # Track game variables
        self.current_distance = 0.0
        self.previous_distance = 0
        self.logger = logger  # Allow logger to be assigned dynamically
        self.episode_count = 0

        self.statuses = {0: "WAITING", 1: "RUNNING", 2: "JUMPING", 3: "DUCKING", 4: "CRASHED"}
        self.legal_actions = {
            0: ["jump"],
            1: ["run", "jump", "duck"],
            2: ["run", "fall"],
            3: ["run", "duck", "stand"],
            4: []  # No legal actions when crashed
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.start_game()
        self.current_distance = 0.0  # Reset distance
        self.previous_distance = 0.0
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        # Get the observation before performing the action
        original_observation = self._get_observation()
        current_status = original_observation["status"]
        self.current_distance = original_observation["distance"]
        terminated = self.statuses[original_observation["status"]] == "CRASHED"

        # Check for early termination (crashed status before taking any action)
        if terminated:
            reward = -100.0
            truncated = False
            self.episode_count += 1
            print(f"Game over. Status: CRASHED, Reward: {reward}")
            if self.logger:  # Ensure logger is set
                self.logger.record("rollout/ep_distance", self.current_distance)
                self.logger.dump(step=self.episode_count)  # Ensure logs are written to TensorBoard
            return original_observation, reward, terminated, truncated, {}

        action_str = ["run", "jump", "duck", "fall", "stand"][action]
        
        # Check if the action is legal
        if action_str not in self.legal_actions[current_status]:
            reward = -5.0  # Penalty for illegal action
            
            print(f'Status: {self.statuses[current_status]} | Action: {action_str} | Reward: {reward} | Illegal action')
            truncated = False
            time.sleep(0.1) # sleep for a short duration to simulate action
            new_observation = self._get_observation()
            self.current_distance = float(new_observation["distance"][0])
            terminated = self.statuses[new_observation["status"]] == "CRASHED"
            if terminated and self.logger:
                self.episode_count += 1
                self.logger.record("rollout/ep_distance", self.current_distance)
                self.logger.dump(step=self.episode_count)  # Ensure logs are written to TensorBoard
            return original_observation, reward, terminated, truncated, {}

        # Perform the action and get the new observation
        self.game.send_action(action_str)

        time.sleep(0.1) # sleep for a short duration to allow the action to take effect

        new_observation = self._get_observation()
        self.current_distance = float(new_observation["distance"][0])

        # Check for termination after the action
        terminated = self.statuses[new_observation["status"]] == "CRASHED"

        reward = self._compute_reward(new_observation)
        print(f'Status: {self.statuses[current_status]} | Action: {action_str} | Reward: {reward}')

        truncated = False
        if terminated and self.logger:
            self.episode_count += 1
            self.logger.record("rollout/ep_distance", self.current_distance)
            self.logger.dump(step=self.episode_count)  # Ensure logs are written to TensorBoard

        # Return the original observation (before the action)
        return original_observation, reward, terminated, truncated, {}

    def _get_observation(self):
        state = self.game.get_game_state()
        if state is None:
            return self.observation_space.sample()  # Handle edge case gracefully

        return {
            "status": state["status"],
            "distance": np.array([state["distance"]], dtype=np.float32),
            "speed": np.array([state["speed"] / 10.0], dtype=np.float32),
            "jump_velocity": np.array([state["jump_velocity"] / 50.0], dtype=np.float32),
            "y_position": np.array([state["y_position"] / 100.0], dtype=np.float32),
            "obstacles": np.array(state["obstacles"], dtype=np.float32)
        }

    def _compute_reward(self, observation):
        
        # Immediate penalty for crashing
        if observation["status"] == 4:
            print(f"Game over. Status: CRASHED, Reward: {-100.0}")
            return -100.0  

        # Survival reward
        reward = 1.0

        # Distance reward
        current_distance = float(observation["distance"][0])
        reward += current_distance - self.previous_distance
        self.previous_distance = current_distance

        # Penalty for being in a non-running state
        if self.statuses[observation["status"]] != "RUNNING":
            reward -= 0.1

        return round(reward, 2)

