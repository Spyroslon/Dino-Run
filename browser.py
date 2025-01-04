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
            self.page.goto('http://example.com') # Try to navigate to Google to trigger the offline dinosaur game
        except:
            pass # This error is expected since we're offline
        time.sleep(1)  # Wait for the game to load
        self.page.keyboard.press('Space')  # Start the game with the spacebar
        print("Game started!")

    def get_game_state(self):
        """Fetch important game state parameters."""
        return {
            "status": self.page.evaluate("() => Runner.instance_.tRex.status"),
            "distance": self.page.evaluate("() => Runner.instance_.distanceMeter.digits.join('')"),
            "speed": self.page.evaluate("() => Runner.instance_.currentSpeed"),
            "jump_velocity": self.page.evaluate("() => Runner.instance_.tRex.jumpVelocity"),
            "y_position": self.page.evaluate("() => Runner.instance_.tRex.yPos"),
            "obstacles": self.page.evaluate("() => Runner.instance_.horizon.obstacles.map(o => ({ x: o.xPos, y: o.yPos, width: o.width, height: o.height }))")
        }
    
    def send_action(self, action):
        """Send the specified action to the game."""
        if action == "jump":
            self.page.keyboard.press("Space")
        elif action == "duck":
            self.page.keyboard.down("ArrowDown")
        else:
            self.page.keyboard.up("ArrowDown")
    
    def close(self):
        """Close the browser session gracefully."""
        self.browser.close()
        self.playwright.stop()
    
    def run(self):
        """Keep the game session active without blocking."""
        # Open game and start it
        self.start_game()\

        # Variables for FPS measurement
        frame_count = 0
        start_time = time.time()

        # Keep the game running and periodically check for state
        while True:
            frame_count += 1

            game_state = self.get_game_state()
            print(game_state)  # Log the current game state

            # Example: Send a random action (or based on some logic)
            self.send_action("jump")
            # Calculate FPS (frames per second)
            elapsed_time = time.time() - start_time
            if elapsed_time >= 1.0:  # Calculate FPS every second
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()  # Reset the timer for the next second
            
            time.sleep(.05)  # Adjust the speed of interactions as necessary

# Initialize and run the game
game = DinoGame()
game.run()  # This will continuously interact with the game

# with sync_playwright() as p:
#     browser = p.chromium.launch(headless=False)
#     context = browser.new_context(offline=True)
#     page = context.new_page()

#     try:
#         page.goto('http://example.com') # Try to navigate to Google to trigger the offline dinosaur game
#     except:
#         pass # This error is expected since we're offline

#     time.sleep(2) # Wait for the game to be ready
#     # Start the game with spacebar
#     page.keyboard.press('Space')
#     print("Game started!")
#     # time.sleep(5)
#     page.wait_for_event("close" , timeout = 0)
#     print('after')
#     # after sleep finishes the page closes automatically
