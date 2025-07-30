from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
from dino_env import DinoEnv

# --- CONFIGURATION ---
ALGO = 'dqn'        # 'ppo', 'a2c', 'dqn'
N_ENVS = 1          # Parallel envs for PPO/A2C
SEED = 42
BATCH_SIZE = 1024   # PPO only

if ALGO == 'ppo':
    N_STEPS = 2048
    TOTAL_TIMESTEPS = 1_000_000
elif ALGO == 'a2c':
    N_STEPS = 20
    TOTAL_TIMESTEPS = 300_000
elif ALGO == 'dqn':
    N_STEPS = None
    TOTAL_TIMESTEPS = 500_000
else:
    raise ValueError(f"Unknown algorithm: {ALGO}")
# --- END CONFIGURATION ---

def linear_schedule(initial_value):
    def scheduler(progress_remaining):
        return progress_remaining * initial_value
    return scheduler

# Clean model name for DQN
if ALGO == 'dqn':
    model_name = f"{ALGO}_{N_ENVS}env"
else:
    model_name = f"{ALGO}_{N_ENVS}env_{N_STEPS}steps"
tensorboard_log = f"./tensorboard_logs/{model_name}/"
checkpoint_path = f"./checkpoints/{model_name}/"

env_kwargs = {'verbose': False, 'max_steps': 1000}  # Removed 'seed'

if ALGO in ['ppo', 'a2c']:
    env = make_vec_env(
        DinoEnv,
        n_envs=N_ENVS,
        env_kwargs=env_kwargs,
        seed=SEED
    )
    env = VecMonitor(env)
else:
    env = Monitor(DinoEnv(**env_kwargs))

checkpoint_callback = CheckpointCallback(
    save_freq=25000,
    save_path=checkpoint_path,
    name_prefix=model_name,
)

if ALGO == 'ppo':
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=linear_schedule(5e-4),
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        ent_coef=0.03,
        tensorboard_log=tensorboard_log,
        device='cuda',
        seed=SEED
    )
elif ALGO == 'a2c':
    model = A2C(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=N_STEPS,
        tensorboard_log=tensorboard_log,
        device='cuda',
        seed=SEED
    )
elif ALGO == 'dqn':
    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device='cuda',
        seed=SEED
    )

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        progress_bar=True,
        log_interval=50
    )
    model.save(f"{checkpoint_path}dino_model_{model_name}_final")
except KeyboardInterrupt:
    print("Training interrupted. Saving model...")
    model.save(f"{checkpoint_path}dino_model_{model_name}_INTERRUPTED")
finally:
    env.close()
