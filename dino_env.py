import gymnasium as gym
from gymnasium import spaces
from browser import DinoBrowser

class DinoEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.browser = DinoBrowser()
        self.action_space = spaces.Discrete(3)  # 0: No action, 1: Jump, 2: Duck
        self.observation_space = spaces.Dict({
            "status": spaces.Discrete(3),  # Example: 0: RUNNING, 1: JUMPING, 2: DUCKING
            "distance": spaces.Box(0, float("inf"), shape=(1,)),
            "speed": spaces.Box(0, float("inf"), shape=(1,)),
            "jump_velocity": spaces.Box(-float("inf"), float("inf"), shape=(1,)),
            "y_position": spaces.Box(0, float("inf"), shape=(1,)),
            "obstacles": spaces.Box(-float("inf"), float("inf"), shape=(10, 4)),  # Max 10 obstacles
        })

    def reset(self):
        self.browser.page.reload()
        return self.browser.get_game_state()

    def step(self, action):
        if action == 1:
            self.browser.send_action("jump")
        elif action == 2:
            self.browser.send_action("duck")
        else:
            self.browser.send_action("none")
        state = self.browser.get_game_state()
        reward = 1  # Reward per step; adjust as needed
        done = state["status"] != "RUNNING"
        return state, reward, done, {}

    def close(self):
        self.browser.close()