# Dino-Run

**Reinforcement Learning for Chrome's Dino Game using Playwright, Gymnasium, and Stable Baselines3.**

---

## Requirements

- Python 3.13 (recommended)
- Node.js (for Playwright)
- Google Chrome (for Playwright)

## Setup

```bash
# Create and activate a virtual environment
py -3.13 -m venv .dinorun-venv
source .dinorun-venv/Scripts/activate  # Bash (Windows)

# Install dependencies
pip install -r requirements.txt
playwright install chromium
```

**Download the Dino Game Clone ([wayou/t-rex-runner](https://github.com/wayou/t-rex-runner)) and place it in your project folder as `t-rex-runner/`.**

## Quickstart

1. **Configure training:**
   - Edit `.env.local` to set algorithm, environment count, and hyperparameters.

2. **Train a model:**

   ```bash
   python train.py
   ```

3. **Continue training:**

   ```bash
   python train_continue.py
   ```

4. **Test a model:**

   ```bash
   python test.py
   ```

5. **Monitor training with TensorBoard:**

   ```bash
   tensorboard --logdir=./tensorboard_logs/
   ```

## Notes

- Playwright requires Node.js and will download Chromium on first run.
- All environment logic is in Python; no browser scripting required.
- **Parallel training is limited by Playwright. For best results, use `N_ENVS=1` in `.env.local`.**
- Training configuration is managed via `.env.local` (intended to be shared).

## Files

- `train.py` – Train from scratch
- `train_continue.py` – Continue training
- `train_reinitialize.py` – Reinitialize with new hyperparams
- `test.py` – Run trained model
- `dino_env.py` – Gymnasium environment
- `game.py` – Playwright game interface
- `.env.local` – Training configuration (edit and share)

---

## **Happy running! 🦖**
