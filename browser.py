from playwright.sync_api import sync_playwright

class DinoBrowser:
    def __init__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False)
        self.page = self.browser.new_page()
        self.page.goto("chrome://dino")
    
    def get_game_state(self):
        return {
            "status": self.page.evaluate("() => Runner.instance.tRex.status"),
            "distance": self.page.evaluate("() => Runner.instance.distanceMeter.digits.join('')"),
            "speed": self.page.evaluate("() => Runner.instance.currentSpeed"),
            "jump_velocity": self.page.evaluate("() => Runner.instance.tRex.jumpVelocity"),
            "y_position": self.page.evaluate("() => Runner.instance.tRex.yPos"),
            "obstacles": self.page.evaluate("() => Runner.instance.horizon.obstacles.map(o => ({ x: o.xPos, y: o.yPos, width: o.width, height: o.height }))")
        }
    
    def send_action(self, action):
        if action == "jump":
            self.page.keyboard.press("Space")
        elif action == "duck":
            self.page.keyboard.down("ArrowDown")
        else:
            self.page.keyboard.up("ArrowDown")
    
    def close(self):
        self.browser.close()
        self.playwright.stop()