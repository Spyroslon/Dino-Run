from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from dino_env import DinoEnv

# Load the trained model
model = PPO.load("ppo_3_dino_extended_400000_steps.zip")

# Create a new environment for training
env = DinoEnv()
model.set_env(env)  # Attach the environment to the loaded model

# Define a new checkpoint callback to save further progress
checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path="./checkpoints/",
    name_prefix="ppo_3_dino_extended",
)

# Continue training the model
model.learn(
    total_timesteps=200000,
    callback=checkpoint_callback,
    reset_num_timesteps=False,
)

# Save the updated model
model.save("ppo_3_dino_extended_600000_steps")

# Close the environment
env.close()
