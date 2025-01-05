from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from dino_env import DinoEnv

# Create the environment
env = make_vec_env(DinoEnv, n_envs=1)

# Define a checkpoint callback to save models every 10,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/",
    name_prefix="ppo_dino",
)

# Initialize the PPO model
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
)

# Train the model with the checkpoint callback
model.learn(total_timesteps=100000, callback=checkpoint_callback)

# Save the final model
model.save("dino_model")

# Close the environment
env.close()