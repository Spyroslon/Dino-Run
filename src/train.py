from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
import dino_env  # Registers DinoRun-v0 environment
import gymnasium as gym
from dotenv import load_dotenv
import os

# Load config from .env.local
load_dotenv('.env.local')

# Required environment variables - no defaults
ALGO = os.getenv('ALGO')
N_ENVS = int(os.getenv('N_ENVS'))
SEED = int(os.getenv('SEED'))
TOTAL_TIMESTEPS = int(os.getenv('TOTAL_TIMESTEPS'))
MAX_STEPS = int(os.getenv('MAX_STEPS'))
VERBOSE = int(os.getenv('VERBOSE'))
DEVICE = os.getenv('DEVICE')
LOG_INTERVAL = int(os.getenv('LOG_INTERVAL'))
HEADLESS = int(os.getenv('HEADLESS'))

# Algorithm-specific parameters
if ALGO == 'dqn':
    N_STEPS = None
    BATCH_SIZE = None
    ENT_COEF = None
    model_name = f"{ALGO}_{N_ENVS}env"
else:
    N_STEPS = int(os.getenv('N_STEPS'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
    ENT_COEF = float(os.getenv('ENT_COEF'))
    model_name = f"{ALGO}_{N_ENVS}env_{N_STEPS}steps"

# Setup paths
checkpoint_base = f"./checkpoints/{model_name}"
tensorboard_base = f"./tensorboard_logs/{model_name}"
os.makedirs(checkpoint_base, exist_ok=True)

env_kwargs = {'verbose': VERBOSE > 1, 'max_steps': MAX_STEPS, 'headless': bool(HEADLESS)}

# Create environment
if ALGO in ['ppo', 'a2c']:
    env = make_vec_env('DinoRun-v0', n_envs=N_ENVS, env_kwargs=env_kwargs, seed=SEED)
    env = VecMonitor(env)
else:
    env = Monitor(gym.make('DinoRun-v0', **env_kwargs))

# Get run number using TensorBoard's exact logic - find highest existing number + 1
if os.path.exists(tensorboard_base):
    existing_runs = [d for d in os.listdir(tensorboard_base) if d.startswith(ALGO.upper() + "_")]
    if existing_runs:
        # Extract numbers and find the highest
        numbers = []
        for run in existing_runs:
            try:
                num = int(run.split("_")[1])
                numbers.append(num)
            except (IndexError, ValueError):
                continue
        run_number = max(numbers) + 1 if numbers else 0
    else:
        run_number = 0
else:
    run_number = 0

run_folder = f"{ALGO.upper()}_{run_number}"
checkpoint_path = os.path.join(checkpoint_base, run_folder)
os.makedirs(checkpoint_path, exist_ok=True)

# Create model with tensorboard logging
if ALGO == 'ppo':
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=VERBOSE,
        learning_rate=5e-4,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        ent_coef=ENT_COEF,
        tensorboard_log=tensorboard_base,
        device=DEVICE,
        seed=SEED
    )
elif ALGO == 'a2c':
    model = A2C(
        "MultiInputPolicy",
        env,
        verbose=VERBOSE,
        n_steps=N_STEPS,
        tensorboard_log=tensorboard_base,
        device=DEVICE,
        seed=SEED
    )
elif ALGO == 'dqn':
    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=VERBOSE,
        tensorboard_log=tensorboard_base,
        device=DEVICE,
        seed=SEED
    )
else:
    raise ValueError(f"Unknown algorithm: {ALGO}")

# Setup callbacks with matching path
checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path=checkpoint_path,
    name_prefix=run_folder,
)

# Train model
print(f"Starting {ALGO.upper()} training with {N_ENVS} environments...")
print(f"Total timesteps: {TOTAL_TIMESTEPS:,}")
print(f"Checkpoint path: {checkpoint_path}")

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
        log_interval=LOG_INTERVAL
    )
    model.save(f"{checkpoint_path}/final_model")
    print(f"Training complete. Model saved to {checkpoint_path}/final_model")
except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    model.save(f"{checkpoint_path}/interrupted_model")
finally:
    env.close()
