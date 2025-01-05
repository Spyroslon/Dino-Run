# todo: add legal actions, if not legal action punish and dont do action
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple
from game import DinoGame
import time

class DinoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = DinoGame()
        
        self.action_space = spaces.Discrete(5)
        
        HIGH_BOUND = 1e6
        
        self.observation_space = spaces.Dict({
            "status": spaces.Discrete(5),
            "distance": spaces.Box(low=0, high=HIGH_BOUND, shape=(1,), dtype=np.float32),
            "speed": spaces.Box(low=0, high=1000.0, shape=(1,), dtype=np.float32),
            "jump_velocity": spaces.Box(low=-1000.0, high=1000.0, shape=(1,), dtype=np.float32),
            "y_position": spaces.Box(low=0, high=1000.0, shape=(1,), dtype=np.float32),
            "next_obstacle": spaces.Box(low=-1.0, high=HIGH_BOUND, shape=(4,), dtype=np.float32)
        })
        
        self.current_score = 0
        self.previous_distance = 0
        self.is_jumping = False
        self.is_ducking = False
        self.retry_count = 0
        self.max_retries = 3

    def _safe_get_state(self):
        """Safely get game state with retries"""
        state = self.game.get_game_state()
        # print(state)
        retry_count = 0
        
        while state is None and retry_count < self.max_retries:
            print(f"Got None state, retrying... (attempt {retry_count + 1}/{self.max_retries})")
            time.sleep(0.1)  # Short delay before retry
            state = self.game.get_game_state()
            retry_count += 1
            
        if state is None:
            print("Failed to get valid state after retries, resetting game...")
            self.game.start_game()
            time.sleep(0.5)  # Give the game time to initialize
            state = self.game.get_game_state()
            
        return state

    def _safe_float(self, value, default=0.0, name="unnamed"):
        """Safely convert value to float with error reporting"""
        if value is None:
            # print(f"Warning: {name} is None, using default value {default}")
            return default
        try:
            return float(value)
        except (TypeError, ValueError) as e:
            print(f"Error converting {name} to float: {value}, using default {default}")
            return default

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        self.game.start_game()
        time.sleep(0.5)  # Give the game time to initialize
        self.current_score = 0
        self.previous_distance = 0
        self.is_jumping = False
        self.is_ducking = False
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        current_obs = self._get_observation()
        current_status = current_obs["status"]
        
        # Track if action was actually performed
        action_performed = False
        
        if current_status == 2:
            self.is_jumping = True
        elif current_status == 1:
            self.is_jumping = False
            self.is_ducking = False
        elif current_status == 3:
            self.is_ducking = True

        if action == 1:  # Jump
            if not self.is_ducking:
                self.game.send_action("jump")
                self.is_jumping = True
                action_performed = True
        elif action == 2:  # Duck
            if not self.is_jumping:
                self.game.send_action("duck")
                self.is_ducking = True
                action_performed = True
        elif action == 3:  # Fall
            if self.is_jumping:
                self.game.send_action("fall")
                self.is_jumping = False
                action_performed = True
        elif action == 4:  # Stand
            if self.is_ducking:
                self.game.send_action("stand")
                self.is_ducking = False
                action_performed = True
        else:  # Run
            self.game.send_action("run")

        # time.sleep(0.001)

        # print(current_obs, action)
        observation = self._get_observation()
        reward = self._compute_reward(observation, action_performed)
        terminated = observation["status"] == 4
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        state = self._safe_get_state()
        
        default_obs = {
            "status": 0,
            "distance": np.array([0.0], dtype=np.float32),
            "speed": np.array([0.0], dtype=np.float32),
            "jump_velocity": np.array([0.0], dtype=np.float32),
            "y_position": np.array([0.0], dtype=np.float32),
            "next_obstacle": np.zeros(4, dtype=np.float32)
        }
        
        if state is None:
            return default_obs

        # Get and validate each value individually
        distance = self._safe_float(state.get("distance"), 0.0, "distance")
        speed = self._safe_float(state.get("speed"), 0.0, "speed")
        jump_velocity = self._safe_float(state.get("jump_velocity"), 0.0, "jump_velocity")
        y_position = self._safe_float(state.get("y_position"), 0.0, "y_position")
        
        next_obstacle = np.zeros(4, dtype=np.float32)
        obstacles = state.get("obstacles", [])
        if obstacles and len(obstacles) > 0:
            closest = obstacles[0]
            next_obstacle = np.array([
                self._safe_float(closest.get("x"), 0.0, "obstacle_x"),
                self._safe_float(closest.get("y"), 0.0, "obstacle_y"),
                self._safe_float(closest.get("width"), 0.0, "obstacle_width"),
                self._safe_float(closest.get("height"), 0.0, "obstacle_height")
            ], dtype=np.float32)

        status_map = {
            'WAITING': 0,
            'RUNNING': 1,
            'JUMPING': 2,
            'DUCKING': 3,
            'CRASHED': 4
        }
        
        return {
            "status": status_map.get(state.get("status", "WAITING"), 0),
            "distance": np.array([distance], dtype=np.float32),
            "speed": np.array([speed], dtype=np.float32),
            "jump_velocity": np.array([jump_velocity], dtype=np.float32),
            "y_position": np.array([y_position], dtype=np.float32),
            "next_obstacle": next_obstacle
        }

    def _compute_reward(self, observation: Dict[str, np.ndarray], action_performed: bool) -> float:
        # Large negative reward for crashing
        if observation["status"] == 4:
            return -100.0  # Big penalty for crashing
        
        # Calculate distance reward
        current_distance = float(observation["distance"][0])
        distance_reward = current_distance - self.previous_distance
        self.previous_distance = current_distance
        
        # Base survival reward
        survival_reward = 1.0 if observation["status"] != 4 else 0.0
        
        # Action penalty - adjust to a smaller value
        action_penalty = -1 if action_performed else 0.0
        
        # Small penalty for being in non-running states (but not as severe)
        state_penalty = -0.1 if observation["status"] != 1 else 0.0
        
        total_reward = distance_reward + survival_reward + action_penalty + state_penalty
        
        return float(total_reward)

    def close(self):
        if self.game:
            self.game.close()