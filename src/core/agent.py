# basic agent class used for both the predator and prey
from dataclasses import dataclass
import numpy as np

@dataclass
class Agent:
    # each agent has a position (x, y), movement speed, and radius (for collisions)
    x: float
    y: float
    speed: float = 1.0
    radius: float = 5.0  # for collisions

    def position(self):
        return np.array([self.x, self.y])

    def move(self, dx, dy, bounds):
        # update position given (dx, dy) movement values, 
        # ensures the agent stays within world boundaries
        self.x = np.clip(self.x + dx, 0, bounds[0])
        self.y = np.clip(self.y + dy, 0, bounds[1])