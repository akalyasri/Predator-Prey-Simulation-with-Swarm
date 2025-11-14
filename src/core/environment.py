# 2D world for the predator and prey - goal is to model movement, distance calculation, and capture detection
import numpy as np
from .agent import Agent

from dataclasses import dataclass
from typing import Tuple, List, Dict

class Obstacle:
    def __init__(self, x, y, radius):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)

class Environment:

    def __init__(self, width=100, height=100, capture_radius=5.0, num_obstacles=3, min_r=4.0, max_r=12.0):
        self.width = width
        self.height = height
        self.capture_radius = capture_radius

        self.predator = None
        self.prey = None

        self.num_obstacles = num_obstacles
        self.min_r = min_r
        self.max_r = max_r
        self.obstacles = []

        


    def generate_obstacles(self):

        self.obstacles = []

        for _ in range(self.num_obstacles):

            w = np.random.uniform(5, 15)
            h = np.random.uniform(5, 15)
            x = np.random.uniform(0, self.width - w)
            y = np.random.uniform(0, self.height - h)
            self.obstacles.append((x, y, w, h))

    def reset(self):

        #self.generate_obstacles()

        self.obstacles = []
    
        for _ in range(self.num_obstacles):
            # random radius
            r = np.random.uniform(self.min_r, self.max_r)

            # position must stay inside walls with margin = radius
            x = np.random.uniform(r, self.width - r)
            y = np.random.uniform(r, self.height - r)

            self.obstacles.append(Obstacle(x, y, r))


        # place predator and prey randomly outside obstacles
        while True:
            px = np.random.uniform(0, self.width)
            py = np.random.uniform(0, self.height)
            if not self.collides_with_obstacle(px, py):
                break

        self.predator = Agent(px, py, speed=2.0)

        while True:
            qx = np.random.uniform(0, self.width)
            qy = np.random.uniform(0, self.height)
            if not self.collides_with_obstacle(qx, qy):
                break

        self.prey = Agent(qx, qy, speed=1.8)

    
            
        return self.observe()

    def distance(self):
        # return Euclidean distance between predator and prey
        return np.linalg.norm(self.predator.position() - self.prey.position())

    
    def dist(self, x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


    def nearest_obstacle_dir(self, agent):
        if not self.obstacles:
            return 0.0, 0.0

        ax, ay = agent.position()

        # find closest obstacle center
        best_ob = None
        best_d = float("inf")

        for ob in self.obstacles:
            d = self.dist(ax, ay, ob.x, ob.y)
            if d < best_d:
                best_d = d
                best_ob = ob

        dx = best_ob.x - ax
        dy = best_ob.y - ay
        norm = np.sqrt(dx*dx + dy*dy)

        if norm == 0:
            return 0.0, 0.0

        return dx / norm, dy / norm

    
    def collides_with_obstacle(self, x, y, agent_radius=1.0):
        for ob in self.obstacles:
            d = self.dist(x, y, ob.x, ob.y)
            if d < (agent_radius + ob.radius):
                return True
        return False

    def observe(self):
        # create observation vectors for both predator and prey. these observations will later become NEAT inputs

        rel = self.prey.position() - self.predator.position()
        d = self.distance()

        # normalize relative direction
        norm = np.linalg.norm(rel)
        if norm > 0:
            dx = rel[0] / norm
            dy = rel[1] / norm
        else:
            dx = dy = 0.0

        # nearest obstacle directions
        ob_dx_pred, ob_dy_pred = self.nearest_obstacle_dir(self.predator)
        ob_dx_prey, ob_dy_prey = self.nearest_obstacle_dir(self.prey)

        # for now, predator sees (dx, dy, distance, bias), prey sees (-dx, -dy, distance, bias)
        #pred_obs = np.array([rel[0], rel[1], self.distance(), 1.0])
        #prey_obs = np.array([-rel[0], -rel[1], self.distance(), 1.0])

        # final observation size = 7 for each agent
        pred_obs = np.array([
            dx, dy,          # dir to prey
            d / 150.0,       # scaled distance
            1.0,             # bias
            ob_dx_pred, ob_dy_pred  
        ])

        prey_obs = np.array([
            -dx, -dy,        # dir away from predator
            d / 150.0,
            1.0,
            ob_dx_prey, ob_dy_prey  
        ])
        
        return pred_obs, prey_obs
    
    def obstacle_hide_direction(self, agent, predator):

        if not self.obstacles:
            return 0.0, 0.0

        ax, ay = agent.position()
        px, py = predator.position()

        # find nearest obstacle
        best_ob = None
        best_d = float("inf")

        for ob in self.obstacles:
            d = self.dist(ax, ay, ob.x, ob.y)
            if d < best_d:
                best_d = d
                best_ob = ob

        # vector from predator -> obstacle (so prey tries to put obstacle between them)
        obx, oby = best_ob.x, best_ob.y
        hide_vec = np.array([obx - px, oby - py])
        norm = np.linalg.norm(hide_vec)

        if norm == 0:
            return 0.0, 0.0

        return hide_vec[0] / norm, hide_vec[1] / norm


    def step(self, pred_action, prey_action):
        # update the world by applying both agents actions, each action is a tuple (vx, vy)

        
        # try move predator
        px, py = self.predator.position()
        new_px = px + pred_action[0]
        new_py = py + pred_action[1]

        if not self.collides_with_obstacle(new_px, new_py):
            self.predator.move(*pred_action, bounds=(self.width, self.height))

        # try move prey
        qx, qy = self.prey.position()
        new_qx = qx + prey_action[0]
        new_qy = qy + prey_action[1]

        if not self.collides_with_obstacle(new_qx, new_qy):
            self.prey.move(*prey_action, bounds=(self.width, self.height))

        
        dist = self.distance()
        captured = dist < self.capture_radius
        
        # returns new observations and info dict with distance and capture status
        return self.observe(), {"distance": dist, 
                                "captured": captured,
                                "positions": (self.predator.position(), self.prey.position())
                                }
    
