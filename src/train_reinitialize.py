from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import dino_env  # This registers the environment
import gymnasium as gym

def linear_schedule(initial_value):
    def scheduler(progress_remaining):
        return progress_remaining * initial_value
    return scheduler

# Load the trained model
old_model = PPO.load("ppo_4_dino_400000_steps.zip")

# Create a new environment
env = gym.make('DinoRun-v0')

# Define new hyperparameters for exploration
new_hyperparams = {
    "learning_rate": linear_schedule(5e-4),     # Slightly higher for better fine-tuning
    "n_steps": 2048,                            # Moderate steps for frequent updates
    "batch_size": 512,                          # Ensure batch size aligns with n_steps
    "ent_coef": 0.03,                           # Encourage more exploration
}

# Initialize a new model with updated hyperparameters but reuse the policy
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=new_hyperparams["learning_rate"],
    n_steps=new_hyperparams["n_steps"],
    batch_size=new_hyperparams["batch_size"],
    ent_coef=new_hyperparams["ent_coef"],
    tensorboard_log="./tensorboard_logs/",  # Log to the same folder
    policy_kwargs=old_model.policy_kwargs,  # Keep the same policy architecture
)

# Transfer the weights from the old model
model.policy.load_state_dict(old_model.policy.state_dict())

# Define a checkpoint callback to save progress
checkpoint_callback = CheckpointCallback(
    save_freq=20000,
    save_path="./checkpoints/",
    name_prefix="ppo_5_dino",
)

# Train the model
model.learn(
    total_timesteps=300000,  # Train for more steps
    callback=checkpoint_callback,
    reset_num_timesteps=True,  # Start new step count for clarity
)

# Save the updated model
model.save("ppo_5_dino_200000_steps")

# Close the environment
env.close()
