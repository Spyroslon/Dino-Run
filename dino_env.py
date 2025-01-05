import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import DinoGame
import time

class DinoEnv(gym.Env):
    def __init__(self):
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
        self.current_score = 0
        self.previous_distance = 0

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
        self.current_score = 0
        self.previous_distance = 0
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # Get the observation before performing the action
        original_observation = self._get_observation()
        current_status = original_observation["status"]

        # Check for early termination (crashed status before taking any action)
        if self.statuses[current_status] == "CRASHED":
            reward = -100.0
            terminated = True
            truncated = False
            return original_observation, reward, terminated, truncated, {}

        # Check if the action is legal
        action_str = ["run", "jump", "duck", "fall", "stand"][action]
        if action_str not in self.legal_actions[current_status]:
            # Illegal action punishment
            reward = -10.0
            print(f'Status: {self.statuses[current_status]} | Action: {action_str} | Reward: {reward} | Illegal action')
            terminated = False
            truncated = False
            time.sleep(0.25) # sleep for a short duration to simulate action
            return original_observation, reward, terminated, truncated, {}

        # Perform the action and get the new observation
        self.game.send_action(action_str)
        new_observation = self._get_observation()
        reward = self._compute_reward(new_observation)
        print(f'Status: {self.statuses[current_status]} | Action: {action_str} | Reward: {reward}')

        # Check for termination after the action
        terminated = self.statuses[new_observation["status"]] == "CRASHED"
        truncated = False

        # Return the original observation (before the action)
        return original_observation, reward, terminated, truncated, {}

    def _get_observation(self):
        state = self.game.get_game_state()
        if state is None:
            return self.observation_space.sample()  # Handle edge case gracefully

        return {
            "status": state["status"],
            "distance": np.array([state["distance"]], dtype=np.float32),
            "speed": np.array([state["speed"]], dtype=np.float32),
            "jump_velocity": np.array([state["jump_velocity"]], dtype=np.float32),
            "y_position": np.array([state["y_position"]], dtype=np.float32),
            "obstacles": np.array(state["obstacles"], dtype=np.float32)
        }

    def _compute_reward(self, observation):
        
        # Immediate penalty for crashing
        if observation["status"] == 4:
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

