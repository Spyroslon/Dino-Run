from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def act(self, state):
        pass
    
    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass