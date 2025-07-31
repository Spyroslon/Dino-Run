from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
from dino_env import DinoEnv
from dotenv import load_dotenv
import os

# Load config from .env.local (intended to be shared)
load_dotenv('.env.local')

ALGO = os.getenv('ALGO', 'ppo')
N_ENVS = int(os.getenv('N_ENVS', 1))
SEED = int(os.getenv('SEED', 42))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1024))
N_STEPS = int(os.getenv('N_STEPS', 1024)) if os.getenv('N_STEPS') else None
TOTAL_TIMESTEPS = int(os.getenv('TOTAL_TIMESTEPS', 1000000))
ENT_COEF = float(os.getenv('ENT_COEF', 0.03))
MAX_STEPS = int(os.getenv('MAX_STEPS', 1000))
VERBOSE = int(os.getenv('VERBOSE', 1))
DEVICE = os.getenv('DEVICE', 'cuda')
LOG_INTERVAL = os.getenv('LOG_INTERVAL', 1)

if ALGO not in ['ppo', 'a2c', 'dqn']:
    raise ValueError(f"Unknown algorithm: {ALGO}")
if ALGO == 'dqn':
    N_STEPS = None
    BATCH_SIZE = None
    ENT_COEF = None

model_name = f"{ALGO}_{N_ENVS}env" if ALGO == 'dqn' else f"{ALGO}_{N_ENVS}env_{N_STEPS}steps"
tensorboard_log = f"./tensorboard_logs/{model_name}/"
checkpoint_path = f"./checkpoints/{model_name}/"

env_kwargs = {'verbose': False, 'max_steps': MAX_STEPS}

if ALGO in ['ppo', 'a2c']:
    env = make_vec_env(DinoEnv, n_envs=N_ENVS, env_kwargs=env_kwargs, seed=SEED)
    env = VecMonitor(env)
else:
    env = Monitor(DinoEnv(**env_kwargs))

checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path=checkpoint_path,
    name_prefix=model_name,
)

def linear_schedule(initial_value):
    return lambda progress_remaining: progress_remaining * initial_value

if ALGO == 'ppo':
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=VERBOSE,
        learning_rate=linear_schedule(5e-4),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        ent_coef=ENT_COEF,
        tensorboard_log=tensorboard_log,
        device=DEVICE,
        seed=SEED
    )
elif ALGO == 'a2c':
    model = A2C(
        "MultiInputPolicy",
        env,
        verbose=VERBOSE,
        n_steps=N_STEPS,
        tensorboard_log=tensorboard_log,
        device=DEVICE,
        seed=SEED
    )
elif ALGO == 'dqn':
    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=VERBOSE,
        tensorboard_log=tensorboard_log,
        device=DEVICE,
        seed=SEED
    )

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
        log_interval=LOG_INTERVAL
    )
    model.save(f"{checkpoint_path}dino_model_{model_name}_final")
except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    model.save(f"{checkpoint_path}dino_model_{model_name}_INTERRUPTED")
finally:
    env.close()
