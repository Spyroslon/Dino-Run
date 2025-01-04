from stable_baselines3 import PPO
from dino_env import DinoEnv

env = DinoEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("dino_model")
env.close()