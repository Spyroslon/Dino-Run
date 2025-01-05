import gymnasium as gym
from gymnasium import spaces
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

    def reset(self):
        """Reset the game to the initial state."""
        self.game.start_game()
        return self._get_observations()

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
        done = obs["status"] == "CRASHED"
        return obs, reward, done, {}

    def _get_observations(self):
        """Get the current observation from the game."""
        state = self.game.get_game_state()
        if state:
            return {
                "status": state["status"],
                "distance": [state["distance"]],
                "speed": [state["speed"]],
                "jump_velocity": [state["jump_velocity"]],
                "y_position": [state["y_position"]],
                "obstacles": state["obstacles"][:10],  # Cap to max 10 obstacles
            }
        else:
            return self.observation_space.sample()

    def _compute_reward(self, obs):
        """Compute the reward based on the current observation."""
        # Example: Reward based on distance traveled
        return obs["distance"][0]

    def close(self):
        """Close the game environment."""
        self.game.close()
