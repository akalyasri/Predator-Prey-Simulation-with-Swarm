# simple test controllers for verifying fitness functions 
# - NOT part of the final system; just helpers for testing

import numpy as np

def pred_dummy(obs):
    
    dx, dy = obs[0], obs[1]        # direction to prey
    ob_dx, ob_dy = obs[4], obs[5]  # direction to nearest obstacle

    # desired direction toward prey
    chase = np.array([dx, dy])

    avoid_ob = -0.4 * np.array([ob_dx, ob_dy])   # avoid going into obstacle

    v = chase + avoid_ob
    norm = np.linalg.norm(v)
    
    if norm > 0:
        return (v / norm) * 2.0
    else:
        return np.zeros(2)


def prey_dummy(obs):
    dx, dy = obs[0], obs[1]             # direction away from pred
    ob_dx, ob_dy = obs[4], obs[5]       # obstacle direction

    flee = np.array([dx, dy])
    avoid = -np.array([ob_dx, ob_dy])
    jitter = np.random.uniform(-0.2, 0.2, size=2)       # add slight jitter like before

    v = flee + 0.8 * avoid + jitter
    norm = np.linalg.norm(v)

    if norm > 0:
        v = v / norm

    return v * 2.0

# class wrappers for the coevolution code -> allow neat code to instantiate them with speed, etc.
class GreedyPredatorDummy:
    # class wrapper for pred_dummy 
    def __call__(self, obs):
        return pred_dummy(obs)


class GreedyPreyDummy:
    # class wrapper for prey_dummy 

    def __init__(self, env):
        self.env = env

        
    def __call__(self, obs):

        dx = -obs[0]  # direction away from predator
        dy = -obs[1]

        # base flee vector
        flee = np.array([dx, dy])

        # obstacle avoidance
        ob_dx, ob_dy = obs[4], obs[5]
        avoid_ob = -np.array([ob_dx, ob_dy])  # avoid crashing into obstacles

        # try to hide behind obstacle (uses environment helper)
        hide_dx, hide_dy = self.env.obstacle_hide_direction(self.env.prey,
                                                            self.env.predator)
        hide_vec = np.array([hide_dx, hide_dy])

        # wall avoidance
        px, py = self.env.prey.position()
        wall = np.array([
            0.2 if px < 10 else -0.2 if px > self.env.width - 10 else 0,
            0.2 if py < 10 else -0.2 if py > self.env.height - 10 else 0
        ])

        # weighted combination
        v = (
            1.0 * flee +
            0.5 * avoid_ob +
            0.7 * hide_vec +
            0.3 * wall
        )

        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm

        return v * 1.8