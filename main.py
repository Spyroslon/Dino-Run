from env import DinoGameEnv
from agents import DQNAgent
from utils.logger import Logger
from config import Config
import torch
import os

def main():
    # Create directories if they don't exist
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    
    # Initialize environment and agent
    env = DinoGameEnv()
    agent = DQNAgent(
        state_dim=6,
        action_dim=3,
        hidden_size=Config.HIDDEN_SIZE,
        learning_rate=Config.LEARNING_RATE,
        gamma=Config.GAMMA,
        epsilon_start=Config.EPSILON_START,
        epsilon_end=Config.EPSILON_END,
        epsilon_decay=Config.EPSILON_DECAY,
        buffer_size=Config.BUFFER_SIZE,
        batch_size=Config.BATCH_SIZE
    )
    
    # Initialize logger
    logger = Logger(Config.LOG_DIR)
    
    # Training loop
    try:
        for episode in range(Config.EPISODES):
            state, _ = env.reset()
            episode_reward = 0
            steps = 0
            
            for step in range(Config.MAX_STEPS_PER_EPISODE):
                action = agent.act(state)
                next_state, reward, done, _, _ = env.step(action)
                agent.push_to_memory(state, action, reward, next_state, done)
                
                agent.train()
                state = next_state
                episode_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Log episode results
            logger.log_episode(episode, episode_reward, steps, agent.epsilon)
            
            # Save model periodically
            if (episode + 1) % Config.SAVE_INTERVAL == 0:
                agent.save(os.path.join(
                    Config.MODEL_SAVE_DIR,
                    f'dqn_episode_{episode+1}.pth'
                ))
    
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    finally:
        env.close()

if __name__ == "__main__":
    main()