from stable_baselines3 import PPO
from dino_env import DinoEnv

# Load the environment and trained model
env = DinoEnv()
model = PPO.load("ppo_dino_continuing_100000_steps.zip")

# Test the model
obs, _ = env.reset()

# Run the model in the environment
num_episodes = 0
max_episodes = 10  # Maximum number of episodes to run

while num_episodes < max_episodes:
    # Predict action using the trained model
    action, _states = model.predict(obs, deterministic=True)

    # Step through the environment with the chosen action
    obs, reward, done, truncated, info = env.step(action)

    # If the episode ends, reset the environment
    if done:
        num_episodes += 1
        print("Episode finished.")
        print(f'Episode: {num_episodes}, Distance: {obs["distance"][0]*1000}')
        obs, _ = env.reset()

# Close the environment after testing
env.close()
