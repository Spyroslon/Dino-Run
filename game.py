import time
from playwright.sync_api import sync_playwright

class DinoGame:
    def __init__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False, args=["--mute-audio"])
        self.context = self.browser.new_context(offline=True)
        self.page = self.context.new_page()
        self.max_obstacles = 3

    def start_game(self):
        """Navigate to the Dino game and start it."""
        try:
            self.page.goto('https://github.com/Spyroslon')  # Trigger offline dinosaur game
        except:
            pass  # Expected since we're offline
        print('Starting game')
        time.sleep(.5)
        self.page.keyboard.press('Space')  # Start the game
        time.sleep(1)

    def get_game_state(self):
        """Fetch game state parameters."""
        try:
            distance_str = self.page.evaluate("() => Runner.instance_.distanceMeter.digits.join('')")
            distance = float(distance_str) if distance_str != '' else 0.0

            status_map = {
                'WAITING': 0,
                'RUNNING': 1,
                'JUMPING': 2,
                'DUCKING': 3,
                'CRASHED': 4
            }
            
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
                "speed": float(self.page.evaluate("() => Runner.instance_.currentSpeed")),
                "jump_velocity": float(self.page.evaluate("() => Runner.instance_.tRex.jumpVelocity")),
                "y_position": float(self.page.evaluate("() => Runner.instance_.tRex.yPos")),
                "obstacles": obstacles_features
            }

            return state

        except Exception as e:
            print(f"Error fetching game state: {e}")
            return None

    def send_action(self, action):
        """Send a specified action to the game."""
        if action == "run":
            pass  # No action needed for 'run'
        elif action == "jump":
            self.page.keyboard.press("ArrowUp")
        elif action == "fall":
            self.page.keyboard.press("ArrowDown")
        elif action == "duck":
            self.page.evaluate("() => Runner.instance_.tRex.setDuck(true)")
        elif action == "stand":
            self.page.evaluate("() => Runner.instance_.tRex.setDuck(false)")
        else:
            print(f"Unknown action: {action}")

    def close(self):
        """Close the browser session gracefully."""
        self.browser.close()
        self.playwright.stop()
