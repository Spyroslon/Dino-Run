from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from dino_env import DinoEnv

# Create the environment
env = DinoEnv()

# Initialize the model with MultiInputPolicy
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./dino_tensorboard/")

# Train the model
model.learn(total_timesteps=10000)

# Save the model
model.save("dino_model")

# Close the environment
env.close()
