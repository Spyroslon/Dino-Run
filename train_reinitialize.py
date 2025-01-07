from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from dino_env import DinoEnv

def linear_schedule(initial_value):
    def scheduler(progress_remaining):
        return progress_remaining * initial_value
    return scheduler

# Load the trained model
old_model = PPO.load("dino_model_ppo.zip")

# Create a new environment
env = DinoEnv()

# Define new hyperparameters for exploration
new_hyperparams = {
    "learning_rate": linear_schedule(5e-4),  # Slightly higher for better fine-tuning
    "ent_coef": 0.01,       # Encourage more exploration
    "n_steps": 1024,        # Moderate steps for frequent updates
    "batch_size": 256,      # Ensure batch size aligns with n_steps
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
    save_freq=10000,
    save_path="./checkpoints/",
    name_prefix="ppo_dino_reinitialized",
)

# Train the model
model.learn(
    total_timesteps=100000,  # Train for more steps
    callback=checkpoint_callback,
    reset_num_timesteps=True,  # Start new step count for clarity
)

# Save the updated model
model.save("dino_model_ppo_reinitialized")

# Close the environment
env.close()
