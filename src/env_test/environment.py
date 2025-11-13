# 2D world for the predator and prey - goal is to model movement, distance calculation, and capture detection
import numpy as np
from .agent import Agent

class Environment:

    def __init__(self, width=100, height=100, capture_radius=5.0):
        self.width = width
        self.height = height
        self.capture_radius = capture_radius
        self.predator = None
        self.prey = None

    def reset(self):
        # place predator and prey randomly 
        self.predator = Agent(np.random.uniform(0, self.width),
                              np.random.uniform(0, self.height), speed=2.0)
        
        self.prey = Agent(np.random.uniform(0, self.width),
                          np.random.uniform(0, self.height), speed=1.8)
        
        return self.observe()

    def distance(self):
        # return Euclidean distance between predator and prey
        return np.linalg.norm(self.predator.position() - self.prey.position())

    def observe(self):
        # create observation vectors for both predator and prey. these observations will later become NEAT inputs

        rel = self.prey.position() - self.predator.position()
        
        # for now, predator sees (dx, dy, distance, bias), prey sees (-dx, -dy, distance, bias)
        pred_obs = np.array([rel[0], rel[1], self.distance(), 1.0])
        prey_obs = np.array([-rel[0], -rel[1], self.distance(), 1.0])
        
        return pred_obs, prey_obs

    def step(self, pred_action, prey_action):
        # update the world by applying both agents actions, each action is a tuple (vx, vy)

        self.predator.move(*pred_action, bounds=(self.width, self.height))
        self.prey.move(*prey_action, bounds=(self.width, self.height))
        dist = self.distance()
        captured = dist < self.capture_radius
        
        # returns new observations and info dict with distance and capture status
        return self.observe(), {"distance": dist, 
                                "captured": captured,
                                "positions": (self.predator.position(), self.prey.position())
                                }