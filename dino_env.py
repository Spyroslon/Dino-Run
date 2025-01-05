import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import DinoGame

class DinoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.game = DinoGame()
        self.action_space = spaces.Discrete(5)  # 0: No action, 1: Jump, 2: Duck, 3: Fall, 4: Stand
        self.observation_space = spaces.Dict({
            "status": spaces.Discrete(5),  # WAITING, RUNNING, JUMPING, DUCKING, CRASHED
            "distance": spaces.Box(0, float("inf"), shape=(1,), dtype=float),
            "speed": spaces.Box(0, float("inf"), shape=(1,), dtype=float),
            "jump_velocity": spaces.Box(-float("inf"), float("inf"), shape=(1,), dtype=float),
            "y_position": spaces.Box(0, float("inf"), shape=(1,), dtype=float),
            "obstacles": spaces.Box(-float("inf"), float("inf"), shape=(10, 4), dtype=float),  # Max 10 obstacles
        })

    def reset(self, **kwargs):
        """Reset the game to the initial state."""
        self.game.start_game()
        observation = self._get_observations()
        info = {}  # Additional info can be returned here if needed
        return observation, info

    def step(self, action):
        """Take an action and return the environment's response."""
        if action == 1:
            self.game.send_action("jump")
        elif action == 2:
            self.game.send_action("duck")
        elif action == 3:
            self.game.send_action("fall")
        elif action == 4:
            self.game.send_action("stand")
        else:
            self.game.send_action("run")
        
        obs = self._get_observations()
        reward = self._compute_reward(obs)

        # Determine if the episode is done
        done = obs["status"] == 4  # Check if status is 'CRASHED' (integer value 4)

        # Set `terminated` to True if the episode is done
        terminated = done  # Terminated when the game is over
        truncated = False  # No truncation in this environment (unless you add time limits)

        return obs, reward, terminated, truncated, {}

    def _get_observations(self):
        """Get the current observation from the game."""
        state = self.game.get_game_state()
        if state:
            # Map status string to integer
            status_map = {
                'WAITING': 0,
                'RUNNING': 1,
                'JUMPING': 2,
                'DUCKING': 3,
                'CRASHED': 4
            }

            status = status_map.get(state["status"], 0)  # Default to WAITING (0) if status is unknown
            
            # Handle obstacles
            obstacles = state["obstacles"]
            if len(obstacles) > 0:
                # Convert obstacles list to a numpy array
                obstacles = np.array(obstacles, dtype=float)
            else:
                # Create an empty numpy array if no obstacles are present
                obstacles = np.empty((0, 4), dtype=float)
            
            # Pad or truncate obstacles to match shape (10, 4)
            padded_obstacles = np.zeros((10, 4), dtype=float)
            padded_obstacles[:len(obstacles)] = obstacles[:10]

            return {
                "status": status,  # Return status as integer
                "distance": [state["distance"]],
                "speed": [state["speed"]],
                "jump_velocity": [state["jump_velocity"]],
                "y_position": [state["y_position"]],
                "obstacles": padded_obstacles,  # Fixed shape (10, 4)
            }
        else:
            # Return random observation if the state is unavailable
            return self.observation_space.sample()

    def _compute_reward(self, obs):
        """Compute the reward based on the current observation."""
        # Example: Reward based on distance traveled
        return obs["distance"][0]

    def close(self):
        """Close the game environment."""
        self.game.close()
