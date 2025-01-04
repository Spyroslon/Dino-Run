from playwright.sync_api import sync_playwright
import time

def test_dino_game():
    with sync_playwright() as p:
        # Launch chromium browser
        browser = p.chromium.launch(
            headless=False,
        )
        
        # Create a new context with offline mode enabled
        context = browser.new_context(offline=True)
        
        # Create a new page
        page = context.new_page()
        
        try:
            # Try to navigate to Google to trigger the offline dinosaur game
            page.goto('http://example.com')
        except:
            # This error is expected since we're offline
            pass
        
        # Wait for the game to be ready
        time.sleep(2)
        
        # Start the game with spacebar
        page.keyboard.press('Space')
        
        print("Game started!")
        
        # Wait to observe the game
        # time.sleep(5)
        print('initially here')
        # Optional: Print game state
        while True:
            game_state = page.evaluate("""() => {
                const runner = Runner.instance_;
                return {
                    speed: runner.currentSpeed,
                    distance: runner.distanceMeter.getActualDistance(),
                    isJumping: runner.tRex.jumping,
                    yPos: runner.tRex.yPos
                }
            }""")
            print("Game State:", game_state)
            # except Exception as e:
            #     print("Couldn't get game state:", e)
        
        print('then here')

        # Keep the browser open for observation
        time.sleep(5)
        
        # Clean up
        context.close()
        browser.close()

if __name__ == "__main__":
    test_dino_game()