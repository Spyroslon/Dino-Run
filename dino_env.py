# todo: add legal actions, if not legal action punish and dont do action
import gymnasium as gym
from gymnasium import spaces
import time
import numpy as np
from game import DinoGame

class DinoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = DinoGame()
        
        self.action_space = spaces.Discrete(5)
        self.max_obstacles = 3

        self.observation_space = spaces.Dict({
            "status": spaces.Discrete(5),
            "distance": spaces.Box(low=0, high=1e6, shape=(1,), dtype=np.float32),
            "speed": spaces.Box(low=0, high=100.0, shape=(1,), dtype=np.float32),
            "jump_velocity": spaces.Box(low=-50.0, high=50.0, shape=(1,), dtype=np.float32),
            "y_position": spaces.Box(low=0, high=200.0, shape=(1,), dtype=np.float32),
            "obstacles": spaces.Box(low=-100, high=1000.0, shape=(self.max_obstacles * 4,), dtype=np.float32)
        })
        
        self.current_score = 0
        self.previous_distance = 0

        self.is_jumping = False
        self.is_ducking = False
        self.statuses = {"WAITING" : 0, "RUNNING" : 1, "JUMPING" : 2, "DUCKING" : 3, "CRASHED" : 4}
        self.legal_actions = {1 : ["RUN", "JUMP", "DUCK"], 2 : ["RUN", "FALL"], 3 : ["RUN", "STAND"], 4 : []}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Resetting Game
        self.game.start_game()
        self.current_score = 0
        self.previous_distance = 0
        self.is_jumping = False
        self.is_ducking = False
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
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

    def _get_observation(self):
        state = self.game.get_game_state()
        
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
        distance = state.get("distance") or 0.0
        speed = state.get("speed") or 0.0
        jump_velocity = state.get("jump_velocity") or 0.0
        y_position = state.get("y_position") or 0.0
        
        next_obstacle = np.zeros(4, dtype=np.float32)
        obstacles = state.get("obstacles", [])
        if obstacles and len(obstacles) > 0:
            closest = obstacles[0]
            next_obstacle = np.array([
                closest.get("x") or 0.0,
                closest.get("y") or 0.0,
                closest.get("width") or 0.0,
                closest.get("height") or 0.0
            ], dtype=np.float32)

        return {
            "status": state,
            "distance": np.array([distance], dtype=np.float32),
            "speed": np.array([speed], dtype=np.float32),
            "jump_velocity": np.array([jump_velocity], dtype=np.float32),
            "y_position": np.array([y_position], dtype=np.float32),
            "next_obstacle": next_obstacle
        }

    def _compute_reward(self, observation, action_performed):
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
