import time
from playwright.sync_api import sync_playwright

class DinoGame:
    def __init__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False,
                                                        args=[
                                                            "--no-sandbox",           # Disable sandboxing for performance
                                                            "--disable-dev-shm-usage",# Use RAM instead of shared memory
                                                            "--disable-extensions",   # Disable extensions
                                                            "--mute-audio",           # Disable sound
                                                        ])
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
            self.page.bring_to_front()  # Ensures the page stays in focus
            distance_str = self.page.evaluate("() => Runner.instance_.distanceMeter.digits.join('')")
            distance = float(distance_str) if distance_str != '' else 0.0

            status_map = {
                'WAITING': 0,
                'RUNNING': 1,
                'JUMPING': 2,
                'CRASHED': 3
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
                "speed": round(float(self.page.evaluate("() => Runner.instance_.currentSpeed")),2),
                "jump_velocity": round(float(self.page.evaluate("() => Runner.instance_.tRex.jumpVelocity")),2),
                "y_position": round(float(self.page.evaluate("() => Runner.instance_.tRex.yPos")),2),
                "obstacles": obstacles_features
            }
            # print(state)

            return state

        except Exception as e:
            print(f"Error fetching game state: {e}")
            return None

    def precise_sleep(self, duration):
        """Use perf_counter for more precise sleep duration."""
        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            pass

    def send_action(self, action):
        """Send a specified action to the game."""
        if action == "run":
            # time.sleep(0.125) # Sleep to match jumping delay
            self.precise_sleep(0.125) # Adjusted delay for consitent jumping
            pass  # No action needed for 'run'
        elif action == "jump":
            # self.page.keyboard.press("ArrowUp")
            self.page.keyboard.down("ArrowUp")
            # time.sleep(0.125) # Adjusted delay for consitent jumping
            self.precise_sleep(0.125) # Adjusted delay for consitent jumping
            self.page.keyboard.up("ArrowUp")
        else:
            print(f"Unknown action: {action}")

    def close(self):
        """Close the browser session gracefully."""
        self.browser.close()
        self.playwright.stop()
