import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import DinoGame
import time


class DinoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = DinoGame()

        # Action space: 0=run, 1=jump
        self.action_space = spaces.Discrete(2)

        # Observation space
        self.max_obstacles = 3
        self.observation_space = spaces.Dict({
            "status": spaces.Discrete(4),
            "distance": spaces.Box(low=0, high=1e6, shape=(1,), dtype=np.float32),
            "speed": spaces.Box(low=6, high=100.0, shape=(1,), dtype=np.float32),
            "jump_velocity": spaces.Box(low=-50.0, high=50.0, shape=(1,), dtype=np.float32),
            "y_position": spaces.Box(low=0, high=100.0, shape=(1,), dtype=np.float32),
            "obstacles": spaces.Box(low=-100, high=1000.0, shape=(self.max_obstacles * 4,), dtype=np.float32)
        })

        # Track game variables
        self.current_distance = 0.0
        self.best_distance = 0.0
        self.episode_count = 0

        self.actions = ["run", "jump"]
        self.statuses = {0: "WAITING", 1: "RUNNING", 2: "JUMPING", 3: "CRASHED"}

    def reset(self, seed=None, options=None):
        self.episode_count += 1

        # Restarting Chromium every 100 episodes
        if self.episode_count % 100 == 0:
            print("Restarting browser to prevent lag...")
            self.game.close()
            self.game = DinoGame()  # Reinitialize the game

        super().reset(seed=seed)
        self.game.start_game()
        self.current_distance = 0.0
        self.previous_distance = 0.0
        observation = self._get_observation()
        return observation, {}

    def step(self, action):
        # Get the observation before performing the action
        # original_observation = self._get_observation()

        # current_status = original_observation["status"]
        # terminated = self.statuses[original_observation["status"]] == "CRASHED"

        action_str = self.actions[action]
        self.game.send_action(action_str) # Perform the action and get the new observation

        # time.sleep(0.025) # sleep for a short duration to allow the action to take effect

        new_observation = self._get_observation()
        current_status = new_observation["status"]

        self.current_distance = float(new_observation["distance"][0])

        reward = self._compute_reward(new_observation)

        # Check for termination after the action
        terminated = self.statuses[new_observation["status"]] == "CRASHED"

        print(f'Status: {self.statuses[current_status]} | Action: {action_str} | Reward: {reward}')

        if terminated:
            distance = round(self.current_distance*1000)
            print(f"Game over! Total distance: {distance}")
            if distance > self.best_distance:
                self.best_distance = distance
                print(f"New high score: {distance}")

        # Return the original observation (before the action)
        return new_observation, reward, terminated, False, {}

    def _get_observation(self):
        state = self.game.get_game_state()
        if state is None:
            return self.observation_space.sample()

        return {
            "status": state["status"],
            "distance": np.array([state["distance"] / 1000.0], dtype=np.float32),  # Normalize distance
            "speed": np.array([state["speed"] / 10.0], dtype=np.float32),         # Normalize speed
            "jump_velocity": np.array([state["jump_velocity"] / 50.0], dtype=np.float32),
            "y_position": np.array([state["y_position"] / 100.0], dtype=np.float32),
            "obstacles": np.array(state["obstacles"], dtype=np.float32)
        }

    def _compute_reward(self, observation):
        running_survival_reward = 5.0
        jumping_penalty = -1.0
        crash_penalty = -100.0

        reward = running_survival_reward

        if self.statuses[observation["status"]] == "JUMPING":
            reward = jumping_penalty

        if self.statuses[observation["status"]] == "CRASHED":
            reward = crash_penalty

        return round(reward, 2)