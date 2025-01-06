from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from dino_env import DinoEnv

# Create a vectorized environment
env = make_vec_env(DinoEnv, n_envs=1)

# Define a checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path="./checkpoints/",
    name_prefix="ppo_dino",
)

# Initialize the PPO model with updated parameters
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=5e-5,  # Lower learning rate for stable updates
    n_steps=8192,        # Increased n_steps to balance parallel environments
    batch_size=2048,     # Larger batch size for more robust updates
    tensorboard_log="./tensorboard_logs/",
)

# Train the model
model.learn(total_timesteps=500000, callback=checkpoint_callback)

# Save the final model
model.save("dino_model_ppo")

# Close the environment
env.close()
