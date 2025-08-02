import gymnasium as gym
from gymnasium import spaces
import numpy as np
import threading
import queue
from game import DinoGame, start_dino_server, get_browser

class DinoEnv(gym.Env):
    """Custom environment for Chrome Dino game using Playwright automation."""
    metadata = {'render_modes': ['human']}

    def __init__(self, verbose=False, max_steps=1000, headless=True, render_mode=None):
        super().__init__()
        start_dino_server()
        
        # Store render_mode for compatibility with vectorized environments
        self.render_mode = render_mode
        
        self.verbose = verbose
        self.max_steps = max_steps
        self.headless = headless
        
        # Initialize async components in thread
        self._init_game_thread()
        
        # Action space: 0=run, 1=jump
        self.action_space = spaces.Discrete(2)

        # Observation space (bounds match normalized values)
        self.max_obstacles = 3
        self.observation_space = spaces.Dict({
            "status": spaces.Discrete(4),
            "distance": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),         # distance/1000
            "speed": spaces.Box(low=0.06, high=15.0, shape=(1,), dtype=np.float32),        # speed/10 
            "jump_velocity": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32), # velocity/50
            "y_position": spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),       # y_pos/100
            "obstacles": spaces.Box(low=-100, high=1000.0, shape=(self.max_obstacles * 4,), dtype=np.float32)
        })

        # Track game state
        self.current_distance = 0.0
        self.previous_distance = 0.0
        self.best_distance = 0.0
        self.episode_count = 0
        self.current_step = 0

        self.actions = ["run", "jump"]
        self.statuses = {0: "WAITING", 1: "RUNNING", 2: "JUMPING", 3: "CRASHED"}

    def _init_game_thread(self):
        """Initialize game in separate thread to handle async operations."""
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.game_thread = threading.Thread(target=self._game_loop, daemon=True)
        self.game_thread.start()
        
    def _game_loop(self):
        """Main game loop running in separate thread."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        browser = get_browser(headless=self.headless)
        game = DinoGame(browser, verbose=self.verbose)
        
        async def handle_commands():
            await game.init()
            
            while True:
                try:
                    command, args = self.command_queue.get(timeout=0.1)
                    
                    if command == "start_game":
                        await game.start_game()
                        self.result_queue.put("started")
                    elif command == "get_state":
                        state = await game.get_game_state()
                        self.result_queue.put(state)
                    elif command == "action":
                        await game.send_action(args)
                        self.result_queue.put("action_done")
                    elif command == "reset":
                        await game.start_game()
                        self.result_queue.put("reset_done")
                    elif command == "close":
                        await game.close()
                        break
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    self.result_queue.put(f"error: {e}")
                    
        loop.run_until_complete(handle_commands())
        
    def _send_command(self, command, args=None):
        """Send command to game thread and wait for result."""
        self.command_queue.put((command, args))
        try:
            result = self.result_queue.get(timeout=5.0)
            if isinstance(result, str) and result.startswith("error:"):
                raise RuntimeError(result[6:])
            return result
        except queue.Empty:
            raise RuntimeError(f"Command {command} timed out")

    def reset(self, seed=None, options=None):
        """Reset environment and start a new game episode."""
        super().reset(seed=seed)
        
        self.episode_count += 1
        self.current_step = 0
        self.current_distance = 0.0
        self.previous_distance = 0.0
        
        # Start new game
        self._send_command("start_game")
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info

    def step(self, action):
        """Take an action and return the new state, reward, and episode info."""
        self.current_step += 1
        action_str = self.actions[action]
        
        # Send action to game
        try:
            self._send_command("action", action_str)
            observation = self._get_observation()
        except Exception as e:
            if self.verbose:
                print(f"Error in step: {e}")
            # Return proper step format with error state
            observation = self._get_fallback_observation()
            reward = -100.0  # Penalty for error
            terminated = True
            truncated = False
            info = {"error": str(e)}
            return observation, reward, terminated, truncated, info
        
        # Calculate reward and episode status
        current_status = observation["status"]
        self.current_distance = float(observation["distance"][0])
        reward = self._compute_reward(observation)
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
        
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Fetch and normalize the current game state as an observation."""
        state = self._send_command("get_state")
        if state is None:
            return self._get_fallback_observation()
        
        return {
            "status": state["status"],
            "distance": np.array([state["distance"] / 1000.0], dtype=np.float32),
            "speed": np.array([state["speed"] / 10.0], dtype=np.float32),
            "jump_velocity": np.array([state["jump_velocity"] / 50.0], dtype=np.float32),
            "y_position": np.array([state["y_position"] / 100.0], dtype=np.float32),
            "obstacles": np.array(state["obstacles"], dtype=np.float32)
        }
    
    def _get_fallback_observation(self):
        """Return a safe fallback observation when game state is unavailable."""
        return {
            "status": 3,  # CRASHED
            "distance": np.array([0.0], dtype=np.float32),
            "speed": np.array([0.06], dtype=np.float32),  # Minimum speed normalized (6/10)
            "jump_velocity": np.array([0.0], dtype=np.float32),
            "y_position": np.array([0.0], dtype=np.float32),
            "obstacles": np.zeros(self.max_obstacles * 4, dtype=np.float32)
        }
    
    def _get_info(self):
        """Get auxiliary info for debugging."""
        return {
            "distance": self.current_distance,
            "best_distance": self.best_distance,
            "step": self.current_step
        }

    def _compute_reward(self, observation):
        """Simple reward: make progress, don't crash. Let the agent learn everything else."""
        status = self.statuses[observation["status"]]
        
        # Progress is the main reward - more distance = better
        progress = observation["distance"][0] - getattr(self, "previous_distance", 0.0)
        self.previous_distance = observation["distance"][0]
        
        if self.verbose and progress > 0:
            print(f"Progress this step: {progress}")
        
        # Scale progress to reasonable reward range
        progress_reward = progress * 100.0  # Will tune based on actual values
        
        # Crash penalty should require ~5-10 seconds of progress to recover from
        if status == "CRASHED":
            return -50.0  # Will tune based on observed progress rates
        else:
            return progress_reward
    
    def close(self):
        """Clean up resources."""
        try:
            self._send_command("close")
        except:
            pass  # Ignore errors on close


# Register the environment with Gymnasium
gym.register(
    id='DinoRun-v0',
    entry_point=DinoEnv,
    max_episode_steps=1000,
    kwargs={
        'verbose': False,
        'max_steps': 1000,
        'headless': True,
        'render_mode': None
    }
)
