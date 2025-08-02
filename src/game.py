import time
import asyncio
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
from playwright.async_api import async_playwright

# Simple server management
_server_started = False
_server_lock = threading.Lock()

def start_dino_server():
    """Start local HTTP server for Dino game."""
    global _server_started
    with _server_lock:
        if _server_started:
            return
            
        web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 't-rex-runner')
        if not os.path.exists(web_dir):
            raise FileNotFoundError(f"t-rex-runner directory not found at {web_dir}")
        
        handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=web_dir, **kwargs)
        server = HTTPServer(("localhost", 8000), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _server_started = True

class DinoGame:
    """Simple Playwright automation for Chrome Dino game."""
    
    STATUS_MAP = {'WAITING': 0, 'RUNNING': 1, 'JUMPING': 2, 'CRASHED': 3}

    def __init__(self, browser, verbose=False):
        self.browser = browser
        self.verbose = verbose
        self.context = None
        self.page = None

    async def init(self):
        """Initialize browser context and page."""
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()

    async def start_game(self):
        """Start or restart the game."""
        if not self.page:
            raise RuntimeError("Call init() first")
        
        # Check if game is already loaded
        try:
            game_exists = await self.page.evaluate("() => !!Runner.instance_")
        except:
            game_exists = False
        
        if not game_exists:
            # Load game page only if not already loaded
            await self.page.goto('http://localhost:8000', wait_until='domcontentloaded')
            await asyncio.sleep(0.2)
        
        # Start/restart game
        await self.page.keyboard.press('Space')
        
        # Wait for game to actually start running with clean state
        for _ in range(20):  # Max 1 second wait
            try:
                state_check = await self.page.evaluate("""
                    () => {
                        const runner = Runner.instance_;
                        if (!runner || !runner.tRex) return null;
                        return {
                            status: runner.tRex.status,
                            distance: runner.distanceMeter.digits.join(''),
                            crashed: runner.crashed
                        };
                    }
                """)
                
                if (state_check and 
                    state_check['status'] == 'RUNNING' and 
                    not state_check['crashed']):
                    # Additional small delay to ensure clean state
                    await asyncio.sleep(0.05) 
                    break
                    
                await asyncio.sleep(0.05)
            except:
                await asyncio.sleep(0.05)

    async def get_game_state(self):
        """Get current game state."""
        if not self.page:
            return None
            
        try:
            state_data = await self.page.evaluate("""
                () => {
                    const runner = Runner.instance_;
                    if (!runner || !runner.tRex) return null;
                    
                    const distanceStr = runner.distanceMeter.digits.join('');
                    const obstacles = runner.horizon.obstacles.map(o => ({
                        x: o.xPos,
                        y: o.yPos,
                        width: o.width,
                        height: o.typeConfig?.height || 50
                    }));
                                        
                    return {
                        distance: distanceStr,
                        status: runner.tRex.status,
                        speed: runner.currentSpeed,
                        jumpVelocity: runner.tRex.jumpVelocity,
                        yPos: runner.tRex.yPos,
                        obstacles: obstacles,
                        crashed: runner.crashed
                    };
                }
            """)
            
            if not state_data:
                return None
            
            # Pad obstacles to exactly 3
            obstacles = state_data['obstacles']
            while len(obstacles) < 3:
                obstacles.append({"x": 0, "y": 0, "width": 0, "height": 0})
            obstacles = obstacles[:3]  # Take only first 3
            
            # Flatten obstacle data
            obstacle_features = []
            for obs in obstacles:
                obstacle_features.extend([obs["x"], obs["y"], obs["width"], obs["height"]])
            
            distance = float(state_data['distance']) if state_data['distance'] else 0.0
            
            return {
                "status": self.STATUS_MAP[state_data['status']],
                "distance": distance,
                "speed": float(state_data['speed']),
                "jump_velocity": float(state_data['jumpVelocity']),
                "y_position": float(state_data['yPos']),
                "obstacles": obstacle_features
            }
            
        except Exception as e:
            if self.verbose:
                print(f"Error getting game state: {e}")
            return None

    async def send_action(self, action):
        """Send action to game."""
        if not self.page:
            return
            
        try:
            if action == "jump":
                await self.page.keyboard.down("ArrowUp")
                await asyncio.sleep(0.08)  # Fixed jump duration
                await self.page.keyboard.up("ArrowUp")
            # "run" action does nothing (default state)
            
        except Exception as e:
            if self.verbose:
                print(f"Error sending action: {e}")

    async def close(self):
        """Clean up resources."""
        if self.page:
            await self.page.close()
            self.page = None
        if self.context:
            await self.context.close()
            self.context = None

# Simple browser creation for the environment
async def create_browser(headless=True):
    """Create a simple browser instance."""
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(
        headless=headless,
        args=["--no-sandbox", "--disable-dev-shm-usage", "--mute-audio"]
    )
    return playwright, browser

def get_browser(headless=True):
    """Sync wrapper for browser creation."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    playwright, browser = loop.run_until_complete(create_browser(headless))
    return browser

if __name__ == "__main__":
    async def main():
        start_dino_server()
        playwright, browser = await create_browser(headless=False)
        game = DinoGame(browser, verbose=True)
        await game.init()
        await game.start_game()
        
        # Test the game
        for i in range(10):
            state = await game.get_game_state()
            print(f"Step {i}: {state}")
            if i % 3 == 0:
                await game.send_action("jump")
            await asyncio.sleep(0.5)
        
        await game.close()
        await browser.close()
        await playwright.stop()
    
    asyncio.run(main())