# Dino-Run
Reinforcement Learning on Chrome's Dino Run using playwright and openai's gymnasium libraries.

Will be testing different reinforcement learning algorithms.

The different approach from existing solutions is instead of using captured frames to instead use the values stored in Javascript Objects.

Parameters:
- Dino Status
- Distance
- Dino Current Speed
- Dino yPos
- List of Obstacles coming

Setting up Virtual enviornment

1. python -m venv .venv
2. source venv/Scripts/activate  # Windows
3. pip install -r requirements.txt
4. playwright install chromium
