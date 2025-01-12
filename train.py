from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from dino_env import DinoEnv

def linear_schedule(initial_value):
    def scheduler(progress_remaining):
        return progress_remaining * initial_value
    return scheduler

# Create a vectorized environment
env = make_vec_env(DinoEnv, n_envs=1)

# Define a checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path="./checkpoints/",
    name_prefix="ppo_2_dino",
)

# Initialize the PPO model with updated parameters
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=linear_schedule(5e-4),
    n_steps=2048,                           # Reduced for faster updates
    batch_size=512,                         # Ensure batch_size divides n_steps evenly
    ent_coef=0.03,                          # Start with exploration
    tensorboard_log="./tensorboard_logs/",
)

# Train the model
model.learn(total_timesteps=300000, callback=checkpoint_callback)

# Save the final model
model.save("dino_model_ppo_2_final")

# Close the environment
env.close()
