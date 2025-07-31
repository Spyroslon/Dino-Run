import gymnasium as gym
from gymnasium import spaces
import numpy as np
import asyncio
from game import DinoGame, start_dino_server, get_browser
import time

class DinoEnv(gym.Env):
    """
    Gymnasium environment for Chrome Dino game using Playwright automation.
    Handles async game logic and provides RL-friendly API.
    """
    def __init__(self, verbose=False, max_steps=1000, headless=True):
        start_dino_server()
        super().__init__()
        self.verbose = verbose
        self.max_steps = max_steps
        self.headless = headless
        self.browser = get_browser(headless=headless)
        self.game = DinoGame(self.browser, verbose=verbose)
        self._run_async(self.game.init())

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
        self.current_step = 0

        self.actions = ["run", "jump"]
        self.statuses = {0: "WAITING", 1: "RUNNING", 2: "JUMPING", 3: "CRASHED"}

    def _run_async(self, coro):
        """Run async coroutine in the current event loop."""
        return asyncio.get_event_loop().run_until_complete(coro)

    def _reset_game(self):
        """Close and reinitialize the DinoGame instance."""
        self._run_async(self.game.close())
        self.game = DinoGame(self.browser, verbose=self.verbose)
        self._run_async(self.game.init())

    def reset(self, seed=None, options=None):
        """Reset environment and start a new game episode."""
        self.episode_count += 1
        self.current_step = 0
        if self.episode_count % 100 == 0:
            self._reset_game()
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self._run_async(self.game.start_game())
        self.current_distance = 0.0
        self.previous_distance = 0.0
        observation = self._retry_observation(seed)
        return observation, {"seed": seed}

    def _retry_observation(self, seed):
        """Retry observation up to 5 times if initial state is invalid."""
        for retry in range(5):
            obs = self._get_observation()
            if obs is not None and not (isinstance(obs, dict) and obs["status"] == 0 and obs["distance"][0] == 0):
                return obs
            self._reset_game()
            self._run_async(self.game.start_game())
        raise RuntimeError("Failed to reset environment after 5 attempts.")

    def step(self, action):
        """Take an action and return the new state, reward, and episode info."""
        self.current_step += 1
        action_str = self.actions[action]
        try:
            self._run_async(self.game.send_action(action_str))
            new_observation = self._get_observation()
        except Exception as e:
            if self.verbose:
                print(f"Error in step: {e}")
            return self.reset()
        if new_observation is None:
            if self.verbose:
                print("Observation is None, forcing reset.")
            return self.reset()
        current_status = new_observation["status"]
        self.current_distance = float(new_observation["distance"][0])
        reward = self._compute_reward(new_observation)
        terminated = self.statuses[current_status] == "CRASHED"
        truncated = self.current_step >= self.max_steps
        if self.verbose:
            print(f'Status: {self.statuses[current_status]} | Action: {action_str} | Reward: {reward}')
        if terminated:
            distance = round(self.current_distance * 1000)
            if distance > self.best_distance:
                self.best_distance = distance
                if self.verbose:
                    print(f"New high score: {distance}")
        info = {
            "distance": self.current_distance,
            "best_distance": self.best_distance,
            "step": self.current_step
        }
        return new_observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Fetch and normalize the current game state as an observation."""
        state = self._run_async(self.game.get_game_state())
        if state is None:
            return self.observation_space.sample()
        return {
            "status": state["status"],
            "distance": np.array([state["distance"] / 1000.0], dtype=np.float32),  # Normalize distance
            "speed": np.array([state["speed"] / 10.0], dtype=np.float32),          # Normalize speed
            "jump_velocity": np.array([state["jump_velocity"] / 50.0], dtype=np.float32),
            "y_position": np.array([state["y_position"] / 100.0], dtype=np.float32),
            "obstacles": np.array(state["obstacles"], dtype=np.float32)
        }

    def _compute_reward(self, observation):
        """Calculate reward based on game progress and status."""
        running_survival_reward = 5.0
        jumping_penalty = -1.0
        crash_penalty = -100.0

        reward = running_survival_reward
        # Add distance-based reward
        progress = observation["distance"][0] - getattr(self, "previous_distance", 0.0)
        reward += progress * 10.0
        self.previous_distance = observation["distance"][0]
        # Add penalties instead of overwriting
        if self.statuses[observation["status"]] == "JUMPING":
            reward += jumping_penalty
        if self.statuses[observation["status"]] == "CRASHED":
            reward += crash_penalty
        return round(reward, 2)