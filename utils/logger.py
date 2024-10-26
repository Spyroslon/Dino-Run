from torch.utils.tensorboard import SummaryWriter
import logging
import os

class Logger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.setup_logging()
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def log_episode(self, episode, reward, steps, epsilon):
        self.writer.add_scalar('Reward/Episode', reward, episode)
        self.writer.add_scalar('Steps/Episode', steps, episode)
        self.writer.add_scalar('Epsilon/Episode', epsilon, episode)
        logging.info(
            f'Episode {episode} - Reward: {reward:.2f}, '
            f'Steps: {steps}, Epsilon: {epsilon:.2f}'
        )