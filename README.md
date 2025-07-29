# Dino-Run

Reinforcement Learning for Chrome's Dino Game using Playwright and Gymnasium.

## Requirements

- Python 3.12
- Node.js (for Playwright)
- Google Chrome (for Playwright)

## Setup

```bash
py -3.12 -m venv .dinorun-venv
# Bash:
source .dinorun-venv/Scripts/activate
pip install -r requirements.txt
playwright install chromium
```

## Quickstart

1. **Train a model:**

   ```bash
   python train.py
   ```

2. **Continue training:**

   ```bash
   python train_continue.py
   ```

3. **Test a model:**

   ```bash
   python test.py
   ```

## Notes

- Make sure the `checkpoints/` and `tensorboard_logs/` folders exist (create if missing).
- Playwright requires Node.js and will download Chromium on first run.
- All code and environment logic is in Python; no browser automation scripting needed.

## Files

- `train.py` – Train from scratch
- `train_continue.py` – Continue training
- `train_reinitialize.py` – Reinitialize with new hyperparams
- `test.py` – Run trained model
- `dino_env.py` – Gymnasium environment
- `game.py` – Playwright game interface
