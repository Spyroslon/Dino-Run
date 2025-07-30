from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from dino_env import DinoEnv

# --- CONFIGURATION ---
ALGO = 'a2c'  # 'ppo', 'a2c', 'dqn'
N_ENVS = 1    # Parallel envs for PPO/A2C
N_STEPS = 2048
BATCH_SIZE = 1024  # PPO only
SEED = 42
TOTAL_TIMESTEPS = 1_000_000
# --- END CONFIGURATION ---

def linear_schedule(initial_value):
    def scheduler(progress_remaining):
        return progress_remaining * initial_value
    return scheduler

model_name = f"{ALGO}_{N_ENVS}env_{N_STEPS}steps"
tensorboard_log = f"./tensorboard_logs/{model_name}/"
checkpoint_path = f"./checkpoints/{model_name}/"

# Parallel envs for PPO/A2C, single env for DQN
if ALGO in ['ppo', 'a2c']:
    env = make_vec_env(
        DinoEnv,
        n_envs=N_ENVS,
        env_kwargs={'verbose': False, 'max_steps': 1000},
        seed=SEED
    )
else:
    env = DinoEnv(verbose=False, max_steps=1000)

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
else:
    raise ValueError(f"Unknown algorithm: {ALGO}")

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
model.save(f"{checkpoint_path}dino_model_{model_name}_final")
env.close()
