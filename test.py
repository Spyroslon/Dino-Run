from stable_baselines3 import PPO
from dino_env import DinoEnv

# Load the environment and trained model
env = DinoEnv()
model = PPO.load("dino_model")

# Test the model
obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(f"Action: {action}, Reward: {reward}")
    total_reward += reward

print(f"Total Reward: {total_reward}")

env.close()
