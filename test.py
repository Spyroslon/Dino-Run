from stable_baselines3 import PPO
from dino_env import DinoEnv

# Load the environment and trained model
env = DinoEnv()
model = PPO.load("ppo_4_dino_280000_steps.zip")

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
