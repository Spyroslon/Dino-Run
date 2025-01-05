import time
from playwright.sync_api import sync_playwright

class DinoGame:
    def __init__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False)
        self.context = self.browser.new_context(offline=True)
        self.page = self.context.new_page()

    def start_game(self):
        """Navigate to the Dino game and start it."""
        try:
            self.page.goto('http://example.com')  # Trigger offline dinosaur game
        except:
            pass  # Expected since we're offline
        time.sleep(1)
        self.page.keyboard.press('Space')  # Start the game
        print("Game started!")

    def get_game_state(self):
        """Fetch game state parameters."""
        try:
            return {
                "status": self.page.evaluate("() => Runner.instance_.tRex.status"),
                "distance": float(self.page.evaluate("() => Runner.instance_.distanceMeter.getActualDistance()")),
                "speed": float(self.page.evaluate("() => Runner.instance_.currentSpeed")),
                "jump_velocity": float(self.page.evaluate("() => Runner.instance_.tRex.jumpVelocity")),
                "y_position": float(self.page.evaluate("() => Runner.instance_.tRex.yPos")),
                "obstacles": self.page.evaluate("() => Runner.instance_.horizon.obstacles.map(o => ({ x: o.xPos, y: o.yPos, width: o.width, height: o.height }))")
            }
        except Exception as e:
            print(f"Error fetching game state: {e}")
            return None

    def send_action(self, action):
        """Send a specified action to the game."""
        if action == "run":
            print("Action: Run")
        elif action == "jump":
            print("Action: Jump")
            self.page.keyboard.press("Space")
        elif action == "fall":
            print("Action: Fall")
            self.page.keyboard.press("ArrowDown")
        elif action == "duck":
            print("Action: Duck")
            self.page.evaluate("() => Runner.instance_.tRex.setDuck(true)")
        elif action == "stand":
            print("Action: Stand")
            self.page.evaluate("() => Runner.instance_.tRex.setDuck(false)")
        else:
            print(f"Invalid action: {action}")

    def close(self):
        """Close the browser session gracefully."""
        self.browser.close()
        self.playwright.stop()

    def run(self):
        """Run the game for testing purposes."""
        self.start_game()
        frame_count = 0
        start_time = time.time()

        while True:
            frame_count += 1
            game_state = self.get_game_state()
            if game_state:
                print(game_state)

            self.send_action("jump")  # Example action

            # Calculate FPS
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()

            time.sleep(0.05)

# Initialize and run the game
game = DinoGame()
game.run()  # This will continuously interact with the game
