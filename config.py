class Config:
    # Environment settings
    CHROME_DRIVER_PATH = None  # Let webdriver_manager handle it
    HEADLESS = False
    
    # Training settings
    EPISODES = 1000
    MAX_STEPS_PER_EPISODE = 10000
    
    # Agent settings
    LEARNING_RATE = 0.00025
    GAMMA = 0.99
    EPSILON_START = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY = 0.995
    
    # Neural Network settings
    HIDDEN_SIZE = 128
    
    # Replay Buffer settings
    BUFFER_SIZE = 100000
    BATCH_SIZE = 32
    
    # Logging settings
    LOG_DIR = "logs"
    MODEL_SAVE_DIR = "saved_models"
    SAVE_INTERVAL = 100