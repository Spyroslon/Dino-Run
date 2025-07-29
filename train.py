from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from dino_env import DinoEnv

def linear_schedule(initial_value):
    def scheduler(progress_remaining):
        return progress_remaining * initial_value
    return scheduler

N_ENVS = 1
N_STEPS = 2048
BATCH_SIZE = 1024
SEED = 42

# Single training environment (headless, fast)
env = make_vec_env(
    DinoEnv,
    n_envs=N_ENVS,
    env_kwargs={'verbose': False, 'max_steps': 1000},
    seed=SEED
)

checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path="./checkpoints/",
    name_prefix="ppo_5_dino",
)

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=linear_schedule(5e-4),
    n_steps=N_STEPS,
    batch_size=BATCH_SIZE,
    ent_coef=0.03,
    tensorboard_log="./tensorboard_logs/",
    device='cuda'
)

model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)
model.save("dino_model_ppo_5_final")
env.close()
