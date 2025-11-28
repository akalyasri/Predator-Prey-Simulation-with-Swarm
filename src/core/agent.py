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

    def move(self, vx, vy, bounds, obstacles=None):

        new_x = self.x + vx
        new_y = self.y + vy

        new_x = np.clip(new_x, 0, bounds[0])
        new_y = np.clip(new_y, 0, bounds[1])

        # if obstacles exist -> prevent movement into them
        #if obstacles:
        #    for (ox, oy, w, h) in obstacles:
    
        #        if ox <= new_x <= ox + w and oy <= new_y <= oy + h:
        #            # cancel movement 
         #           return

        self.x, self.y = new_x, new_y

