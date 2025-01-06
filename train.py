from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from dino_env import DinoEnv

# Define a unified log folder for TensorBoard
log_dir = "./tensorboard_logs/PPO_5"

# Configure the logger to log into the TensorBoard directory
new_logger = configure(log_dir, ["stdout", "tensorboard"])

def make_logged_env():
    return DinoEnv(logger=new_logger)

# Create the environment
env = make_vec_env(make_logged_env, n_envs=1) # consider increasing this

# Define a checkpoint callback to save models every 10,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path="./checkpoints/",
    name_prefix="ppo_dino",
)

# Initialize the PPO model
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    n_steps=4096,
    batch_size=1024, # see how much memory this uses and consider increasing it
    tensorboard_log=None,
)

# Manually set the custom logger for the model
model.set_logger(new_logger)

# Train the model with the checkpoint callback
model.learn(total_timesteps=100000, callback=checkpoint_callback)

# Save the final model
model.save("dino_model")

# Close the environment
env.close()