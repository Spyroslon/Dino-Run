import time
from playwright.async_api import async_playwright
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import asyncio

# --- Shared infrastructure ---
_server_started = False
_shared_browser = None
_shared_playwright = None
_shared_browser_headless = None
_lock = threading.Lock()

def start_dino_server():
    """Start the local HTTP server for the Dino game if not already running."""
    global _server_started
    with _lock:
        if not _server_started:
            web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 't-rex-runner')
            if not os.path.exists(web_dir):
                raise FileNotFoundError(f"t-rex-runner directory not found at {web_dir}")
            
            try:
                handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=web_dir, **kwargs)
                server = HTTPServer(("localhost", 8000), handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                _server_started = True
            except Exception as e:
                raise RuntimeError(f"Failed to start server on port 8000: {e}")

async def get_shared_browser(headless=True):
    """Get the shared browser instance."""
    global _shared_browser, _shared_playwright, _shared_browser_headless
    if _shared_browser is None or _shared_browser_headless != headless:
        if _shared_browser:
            await _shared_browser.close()
        if _shared_playwright:
            await _shared_playwright.stop()
        
        _shared_playwright = await async_playwright().start()
        _shared_browser = await _shared_playwright.chromium.launch(headless=headless, args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-extensions",
            "--mute-audio",
        ])
        _shared_browser_headless = headless
    return _shared_browser

def get_browser(headless=True):
    """Sync wrapper for shared browser."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(get_shared_browser(headless))

class DinoGame:
    """
    Playwright automation for the Chrome Dino game.
    Handles game state, actions, and browser context.
    """
    STATUS_MAP = {
        'WAITING': 0,
        'RUNNING': 1,
        'JUMPING': 2,
        'CRASHED': 3
    }

    def __init__(self, browser, verbose=False):
        self.verbose = verbose
        self.max_obstacles = 3
        self.context = None
        self.page = None
        self.browser = browser

    async def init(self):
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()

    async def start_game(self):
        if not self.page:
            raise RuntimeError("Page not initialized. Call init() first.")
        
        # Check if the game instance exists and get its state
        try:
            game_state = await self.page.evaluate("""
                () => {
                    if (!Runner.instance_) return 'no_instance';
                    if (!Runner.instance_.tRex) return 'no_trex';
                    if (Runner.instance_.crashed) return 'crashed';
                    return 'running';
                }
            """)
        except Exception:
            game_state = 'no_instance'

        if game_state == 'crashed':
            # Game is loaded but crashed - restart and wait for actual running state
            if self.verbose:
                print('Game crashed, restarting...')
            await self.page.keyboard.press('Space')
            # Wait for game to transition from crashed to actually running
            await self._wait_for_running_state()
            # Give game a moment to stabilize before reading states
            await asyncio.sleep(0.1)
        elif game_state in ['running', 'no_trex']:
            # Game is running or loading - don't reload page, just ensure it's started
            if self.verbose:
                print('Game already loaded, ensuring started...')
            try:
                # Press space to start if waiting, or do nothing if already running
                await self.page.keyboard.press('Space')
            except Exception:
                pass
        else:
            # No game instance - need to load the page
            if self.verbose:
                print('Loading game page...')
            await self.page.goto('http://localhost:8000', wait_until='domcontentloaded')
            # Wait for game to initialize
            await self._wait_for_running_state(timeout=2.0)
            await self.page.keyboard.press('Space')

    async def _wait_for_running_state(self, timeout=1.0):
        """Wait for game to be running."""
        start_time = asyncio.get_event_loop().time()
        while True:
            try:
                status = await self.page.evaluate("() => Runner.instance_?.tRex?.status")
                if status == 'RUNNING':
                    await asyncio.sleep(0.1)  # Brief delay to let game fully stabilize
                    return
                    
                if asyncio.get_event_loop().time() - start_time > timeout:
                    return
                    
                await asyncio.sleep(0.05)
            except Exception:
                await asyncio.sleep(0.05)

    async def get_game_state(self):
        if not self.page:
            return None
            
        try:
            await self.page.bring_to_front()
            
            # Check if game instance exists
            game_ready = await self.page.evaluate("() => !!Runner.instance_ && !!Runner.instance_.tRex")
            if not game_ready:
                return None
            
            # Get all game state in one evaluation to reduce async calls
            state_data = await self.page.evaluate("""
                () => {
                    const runner = Runner.instance_;
                    if (!runner || !runner.tRex) return null;
                    
                    const distanceStr = runner.distanceMeter.digits.join('');
                    const obstacles = runner.horizon.obstacles.map(o => {
                        const height = o.typeConfig?.height || 50;
                        console.log(`Obstacle: x=${o.xPos}, y=${o.yPos}, width=${o.width}, height=${height}, type=${o.typeConfig?.type}`);
                        return {
                            x: o.xPos,
                            y: o.yPos,
                            width: o.width,
                            height: height
                        };
                    });
                    
                    return {
                        distance: distanceStr,
                        status: runner.tRex.status,
                        speed: runner.currentSpeed,
                        jumpVelocity: runner.tRex.jumpVelocity,
                        yPos: runner.tRex.yPos,
                        obstacles: obstacles
                    };
                }
            """)
            
            if not state_data:
                return None
            
            # Ensure we have exactly max_obstacles
            obstacles = state_data['obstacles']
            while len(obstacles) < self.max_obstacles:
                obstacles.append({"x": 0, "y": 0, "width": 0, "height": 0})
            obstacles_features = [value for obstacle in obstacles[:self.max_obstacles] for value in [obstacle["x"], obstacle["y"], obstacle["width"], obstacle["height"]]]
            
            distance = float(state_data['distance']) if state_data['distance'] != '' else 0.0
            
            state = {
                "status": self.STATUS_MAP[state_data['status']],
                "distance": distance,
                "speed": round(float(state_data['speed']), 2),
                "jump_velocity": round(float(state_data['jumpVelocity']), 2),
                "y_position": round(float(state_data['yPos']), 2),
                "obstacles": obstacles_features
            }
            return state
        except Exception as e:
            if self.verbose:
                print(f"[DinoGame] Error fetching game state: {e}")
            return None

    async def precise_sleep(self, duration):
        if duration > 0.01:
            await asyncio.sleep(duration)
        else:
            start = time.perf_counter()
            while time.perf_counter() - start < duration:
                pass

    async def send_action(self, action):
        if not self.page:
            raise RuntimeError("Page not initialized")
            
        try:
            # Check if game is ready
            game_ready = await self.page.evaluate("() => !!Runner.instance_ && !!Runner.instance_.tRex")
            if not game_ready:
                return  # Game not ready, ignore action
                
            status = await self.page.evaluate("() => Runner.instance_.tRex.status")
            if action == "run":
                pass
            elif action == "jump":
                if status == "RUNNING":
                    # Fixed jump duration for consistency - same action = same result
                    await self.page.keyboard.down("ArrowUp")
                    await self.precise_sleep(0.08)  # Consistent 80ms
                    await self.page.keyboard.up("ArrowUp")
        except Exception as e:
            if self.verbose:
                print(f"[DinoGame] Error in send_action: {e}")
            raise

    async def close(self):
        """Clean up page and context."""
        try:
            if self.page:
                await self.page.close()
                self.page = None
        except Exception as e:
            if self.verbose:
                print(f"Error closing page: {e}")
        
        try:
            if self.context:
                await self.context.close()
                self.context = None
        except Exception as e:
            if self.verbose:
                print(f"Error closing context: {e}")
    
    async def reset(self):
        """Reset the game without reinitializing browser."""
        await self.start_game()

async def main():
    start_dino_server()
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    game = DinoGame(browser, verbose=True)
    await game.init()
    await game.start_game()
    await asyncio.sleep(10)  # Let the game run for a while
    state = await game.get_game_state()
    print(state)
    await game.close()
    await browser.close()
    await playwright.stop()

if __name__ == "__main__":
    asyncio.run(main())
