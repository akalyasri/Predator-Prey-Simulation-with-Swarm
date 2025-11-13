# simple test controllers for verifying fitness functions 
# - NOT part of the final system; just helpers for testing

import numpy as np

def pred_dummy(obs):
    # predator always moves directly toward prey
    dx, dy = obs[0], obs[1]
    v = np.array([dx, dy])
    if np.linalg.norm(v) > 0:
        v = v / np.linalg.norm(v)
    return v * 2.0   # speed = 2 

def prey_dummy(obs):
    dx, dy = obs[0], obs[1]
    v = -np.array([dx, dy])

    # add slight random jitter so it doesn't go straight into a wall
    # gives agents a more realistic challenge
    v += np.random.uniform(-0.3, 0.3, size = 2)

    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm

    return v * 2.0
