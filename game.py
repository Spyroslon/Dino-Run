import time
from playwright.async_api import async_playwright
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import asyncio

# --- Shared infrastructure ---
_shared_browser = None
_shared_playwright = None
_server_started = False
_shared_browser_headless = None

def start_dino_server():
    """Start the local HTTP server for the Dino game if not already running."""
    global _server_started
    if not _server_started:
        web_dir = os.path.join(os.path.dirname(__file__), 't-rex-runner')
        handler = lambda *args, **kwargs: SimpleHTTPRequestHandler(*args, directory=web_dir, **kwargs)
        server = HTTPServer(("localhost", 8000), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        _server_started = True

async def _get_shared_browser(headless=True):
    """Get or create a shared Playwright browser instance."""
    global _shared_browser, _shared_playwright, _shared_browser_headless
    if _shared_browser is None or _shared_browser_headless != headless:
        if _shared_playwright is not None and _shared_browser is not None:
            await _shared_browser.close()
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
    """Synchronously get the shared browser instance."""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(_get_shared_browser(headless=headless))

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
        await self.page.goto('http://localhost:8000')
        if self.verbose:
            print('Starting game')
        for _ in range(20):
            try:
                if await self.page.evaluate("() => !!Runner.instance_ && !!Runner.instance_.tRex"):
                    break
            except Exception:
                pass
            await asyncio.sleep(0.1)
        await self.page.keyboard.press('Space')
        await asyncio.sleep(0.5)

    async def get_game_state(self):
        try:
            await self.page.bring_to_front()
            distance_str = await self.page.evaluate("() => Runner.instance_.distanceMeter.digits.join('')")
            distance = float(distance_str) if distance_str != '' else 0.0
            status_map = self.STATUS_MAP
            obstacles = await self.page.evaluate("""
                () => Runner.instance_.horizon.obstacles.map(o => ({
                    x: o.xPos,
                    y: o.yPos,
                    width: o.width,
                    height: o.height || 0
                }))
            """)
            while len(obstacles) < self.max_obstacles:
                obstacles.append({"x": 0, "y": 0, "width": 0, "height": 0})
            obstacles_features = [value for obstacle in obstacles[:self.max_obstacles] for value in [obstacle["x"], obstacle["y"], obstacle["width"], obstacle["height"]]]
            state = {
                "status": status_map[await self.page.evaluate("() => Runner.instance_.tRex.status")],
                "distance": distance,
                "speed": round(float(await self.page.evaluate("() => Runner.instance_.currentSpeed")), 2),
                "jump_velocity": round(float(await self.page.evaluate("() => Runner.instance_.tRex.jumpVelocity")), 2),
                "y_position": round(float(await self.page.evaluate("() => Runner.instance_.tRex.yPos")), 2),
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
        try:
            status = await self.page.evaluate("() => Runner.instance_.tRex.status")
            if action == "run":
                pass
            elif action == "jump":
                if status == "RUNNING":
                    await self.page.keyboard.down("ArrowUp")
                    await self.precise_sleep(0.12)
                    await self.page.keyboard.up("ArrowUp")
        except Exception as e:
            if self.verbose:
                print(f"[DinoGame] Error in send_action: {e}")
            raise

    async def close(self):
        if self.page:
            await self.page.close()
        if self.context:
            await self.context.close()

async def main():
    start_dino_server()
    browser = get_browser()
    game = DinoGame(browser, verbose=True)
    await game.init()
    await game.start_game()
    await asyncio.sleep(10)  # Let the game run for a while
    state = await game.get_game_state()
    print(state)
    await game.close()
    await browser.close()
    await _shared_playwright.stop()

if __name__ == "__main__":
    asyncio.run(main())
