import time
from playwright.sync_api import sync_playwright
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

class DinoGame:
    STATUS_MAP = {
        'WAITING': 0,
        'RUNNING': 1,
        'JUMPING': 2,
        'CRASHED': 3
    }

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.server_thread = None
        self._start_dino_server()
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=True,
                                                        args=[
                                                            "--no-sandbox",           # Disable sandboxing for performance
                                                            "--disable-dev-shm-usage",# Use RAM instead of shared memory
                                                            "--disable-extensions",   # Disable extensions
                                                            "--mute-audio",           # Disable sound
                                                        ])
        self.context = self.browser.new_context()  # Removed offline=True
        self.page = self.context.new_page()
        self.max_obstacles = 3

    def _start_dino_server(self):
        """Start a local HTTP server to serve the Dino game."""
        web_dir = os.path.join(os.path.dirname(__file__), 't-rex-runner')
        handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=web_dir, **kwargs)
        server = HTTPServer(("localhost", 8000), handler)
        self.server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        self.server_thread.start()
        if self.verbose:
            print("Started Dino game server at http://localhost:8000")

    def start_game(self):
        """Navigate to the Dino game and start it."""
        self.page.goto('http://localhost:8000')  # Use local server
        if self.verbose:
            print('Starting game')
        # Wait for the tRex to be present (robust)
        for _ in range(20):
            try:
                if self.page.evaluate("() => !!Runner.instance_ && !!Runner.instance_.tRex"):
                    break
            except:
                pass
            time.sleep(0.1)
        self.page.keyboard.press('Space')  # Start the game
        time.sleep(0.5)

    def get_game_state(self):
        """Fetch game state parameters."""
        try:
            self.page.bring_to_front()  # Ensures the page stays in focus
            distance_str = self.page.evaluate("() => Runner.instance_.distanceMeter.digits.join('')")
            distance = float(distance_str) if distance_str != '' else 0.0

            status_map = self.STATUS_MAP
            
            # Get obstacle details (xPos, yPos, width, height)
            obstacles = self.page.evaluate("""
                () => Runner.instance_.horizon.obstacles.map(o => ({
                    x: o.xPos,
                    y: o.yPos,
                    width: o.width,
                    height: o.height || 0
                }))
            """)
            
            # If there are fewer than max_obstacles, pad the data
            while len(obstacles) < self.max_obstacles:
                obstacles.append({"x": 0, "y": 0, "width": 0, "height": 0})

            # Flatten obstacles into a list of features
            obstacles_features = [value for obstacle in obstacles[:self.max_obstacles] for value in [obstacle["x"], obstacle["y"], obstacle["width"], obstacle["height"]]]

            state = {
                "status": status_map[self.page.evaluate("() => Runner.instance_.tRex.status")],
                "distance": distance,
                "speed": round(float(self.page.evaluate("() => Runner.instance_.currentSpeed")),2),
                "jump_velocity": round(float(self.page.evaluate("() => Runner.instance_.tRex.jumpVelocity")),2),
                "y_position": round(float(self.page.evaluate("() => Runner.instance_.tRex.yPos")),2),
                "obstacles": obstacles_features
            }
            # print(state)

            return state

        except Exception as e:
            if getattr(self, "verbose", False):
                print(f"Error fetching game state: {e}")
            return None

    def precise_sleep(self, duration):
        """Use perf_counter for more precise sleep duration."""
        if duration > 0.01:
            time.sleep(duration)
        else:
            start = time.perf_counter()
            while time.perf_counter() - start < duration:
                pass

    def send_action(self, action):
        """Send a specified action to the game."""
        try:
            status = self.page.evaluate("() => Runner.instance_.tRex.status")
            if action == "run":
                pass  # No action needed
            elif action == "jump":
                if status == "RUNNING":  # Only jump if on ground
                    self.page.keyboard.down("ArrowUp")
                    self.precise_sleep(0.12)  # Fixed duration for consistent jump height
                    self.page.keyboard.up("ArrowUp")
            else:
                pass
        except Exception as e:
            if getattr(self, "verbose", False):
                print(f"Error in send_action: {e}")
            raise

    def close(self):
        """Close the browser session gracefully."""
        self.browser.close()
        self.playwright.stop()
