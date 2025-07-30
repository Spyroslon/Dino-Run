# Dino-Run

Reinforcement Learning for Chrome's Dino Game using Playwright and Gymnasium.

## Requirements

- Python 3.10 (recommended)
- Node.js (for Playwright)
- Google Chrome (for Playwright)

## Setup

```bash
py -3.10 -m venv .dinorun-venv
# Bash:
source .dinorun-venv/Scripts/activate
pip install -r requirements.txt
playwright install chromium
```

## Quickstart

1. **Train a model (single environment only):**

   ```bash
   python train.py
   ```

2. **Continue training:**

   ```bash
   python train_continue.py
   ```

3. **Test a model:**

   ```bash
   # Command Prompt or PowerShell
   python test.py
   ```

4. **Monitor training with TensorBoard:**

   ```bash
   tensorboard --logdir=./tensorboard_logs/
   ```

## Notes

- Make sure the `checkpoints/` and `tensorboard_logs/` folders exist (create if missing).
- Playwright requires Node.js and will download Chromium on first run.
- All code and environment logic is in Python; no browser automation scripting needed.
- **Parallel training is not supported due to Playwright limitations. Use only one environment (n_envs=1).**

## Headless Mode Setup

Download the Dino Game Clone (<https://github.com/wayou/t-rex-runner>) and put it in the project folder as `t-rex-runner/`.

## Files

- `train.py` – Train from scratch
- `train_continue.py` – Continue training
- `train_reinitialize.py` – Reinitialize with new hyperparams
- `test.py` – Run trained model
- `dino_env.py` – Gymnasium environment
- `game.py` – Playwright game interface
