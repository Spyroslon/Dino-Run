from stable_baselines3 import PPO
import dino_env  # This registers the environment
import gymnasium as gym
import os

# Load headless setting from environment variable
HEADLESS = int(os.getenv('HEADLESS', 1))  # 1=headless, 0=visual

# Load the environment and trained model
env = gym.make('DinoRun-v0', headless=bool(HEADLESS))  # Enable rendering for visualization
model = PPO.load("checkpoints/ppo_1env_2048steps/dino_model_ppo_1env_2048steps_final.zip")

# Test the model
obs, _ = env.reset()

# Run the model in the environment
num_episodes = 0
max_episodes = 100  # Maximum number of episodes to run
best_distance = 0

while num_episodes < max_episodes:
    # Predict action using the trained model
    action, _states = model.predict(obs, deterministic=True)

    # Step through the environment with the chosen action
    obs, reward, done, truncated, info = env.step(action)

    # If the episode ends, reset the environment
    if done:
        num_episodes += 1
        print("Episode finished.")
        print(f'Episode: {num_episodes}, Distance: {round(obs["distance"][0]*1000)}')
        if round(obs["distance"][0]*1000) > best_distance:
            best_distance = round(obs["distance"][0]*1000)
            print(f'New best distance: {best_distance}')
        obs, _ = env.reset()

print('Testing complete.')
print('Best distance:', best_distance)

# Close the environment after testing
env.close()
