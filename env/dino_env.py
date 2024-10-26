import gymnasium as gym
from gymnasium import spaces
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

class DinoGameEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    def __init__(self):
        super(DinoGameEnv, self).__init__()
        
        # Define action space (0: do nothing, 1: jump, 2: duck)
        self.action_space = spaces.Discrete(3)
        
        # Define observation space (6 game parameters)
        # [status, distance, speed, jump_velocity, y_pos, obstacle_distance]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -20, 0, 0]),
            high=np.array([2, 99999, 50, 20, 200, 1000]),
            dtype=np.float32
        )
        
        # Setup Chrome
        self._setup_chrome()
        
    def _setup_chrome(self):
        """Initialize Chrome browser with the dinosaur game"""
        chrome_options = Options()
        chrome_options.add_argument("--mute-audio")
        # chrome_options.add_argument("--headless")  # Uncomment for headless mode
        
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )
        self.driver.get("chrome://dino")
        
        # Inject helper functions
        self._inject_helper_functions()
        
    def _inject_helper_functions(self):
        """Inject JavaScript functions to help interact with the game"""
        helper_js = """
        function getGameState() {
            const runner = Runner.instance_;
            const tRex = runner.tRex;
            const obstacle = runner.horizon.obstacles[0];
            
            return {
                status: tRex.status,
                distance: runner.distanceMeter.digits.join(''),
                speed: runner.currentSpeed,
                jumpVelocity: tRex.jumpVelocity,
                yPos: tRex.yPos,
                obstacleDistance: obstacle ? 
                    (obstacle.xPos - tRex.xPos) : 
                    999
            };
        }
        
        function isGameOver() {
            return Runner.instance_.crashed;
        }
        
        function startGame() {
            Runner.instance_.play();
        }
        
        function restart() {
            Runner.instance_.restart();
        }
        """
        self.driver.execute_script(helper_js)
        
    def _get_observation(self):
        """Get current game state"""
        game_state = self.driver.execute_script("return getGameState();")
        
        return np.array([
            float(game_state['status']),
            float(game_state['distance']),
            float(game_state['speed']),
            float(game_state['jumpVelocity']),
            float(game_state['yPos']),
            float(game_state['obstacleDistance'])
        ], dtype=np.float32)
    
    def _is_game_over(self):
        """Check if the game is over"""
        return self.driver.execute_script("return isGameOver();")
    
    def step(self, action):
        """Execute action and return new state"""
        # Execute action
        if action == 1:  # Jump
            self.driver.execute_script("Runner.instance_.tRex.startJump();")
        elif action == 2:  # Duck
            self.driver.execute_script("Runner.instance_.tRex.setDuck(true);")
        else:  # Do nothing or stop ducking
            self.driver.execute_script("Runner.instance_.tRex.setDuck(false);")
        
        # Small delay to let the game update
        time.sleep(0.01)
        
        # Get new state
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(observation)
        
        # Check if game is over
        done = self._is_game_over()
        
        return observation, reward, done, False, {}
    
    def _calculate_reward(self, observation):
        """Calculate reward based on game state"""
        if self._is_game_over():
            return -100  # Large penalty for dying
        
        # Base reward for surviving
        reward = 1
        
        # Additional reward based on speed
        reward += observation[2] * 0.1
        
        # Reward for maintaining safe distance from obstacles
        obstacle_distance = observation[5]
        if 100 <= obstacle_distance <= 200:
            reward += 2  # Reward for maintaining safe distance
        elif obstacle_distance < 50:
            reward -= 1  # Penalty for very close obstacles
            
        return reward
    
    def reset(self, seed=None):
        """Reset the game"""
        super().reset(seed=seed)
        
        # Restart the game
        self.driver.execute_script("restart();")
        
        # Small delay to let the game initialize
        time.sleep(0.1)
        
        return self._get_observation(), {}
    
    def close(self):
        """Close the browser"""
        if hasattr(self, 'driver'):
            self.driver.quit()