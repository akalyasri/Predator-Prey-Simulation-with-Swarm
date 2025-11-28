from dataclasses import dataclass
from .environment import Environment
import numpy as np

@dataclass
class EpisodeResult:
    steps: int
    captured: bool
    final_distance: float
    trace: list

def run_episode(pred_controller, prey_controller,
                env=None, T=500, capture_radius=5.0):

    if env is None:
        env = Environment(capture_radius=capture_radius)

    # reset environment
    env.reset()
    trace = []

    # INITIAL OBSERVATION (this MUST exist before loop)
    obs_pred, obs_prey = env.observe()

    for t in range(T):

        # actions based on CURRENT observations
        a_pred = pred_controller(obs_pred)
        a_prey = prey_controller(obs_prey)

        # apply step -> returns NEW observations
        (obs_pred, obs_prey), info = env.step(a_pred, a_prey)

        # record trace
        trace.append({
            "pred_pos": np.array(info["positions"][0], dtype=float),
            "prey_pos": np.array(info["positions"][1], dtype=float),
            "pred_vel": np.array(info["pred_vel"], dtype=float),
            "prey_vel": np.array(info["prey_vel"], dtype=float),
            "distance": float(info["distance"]),
        })

        if info["captured"]:
            return EpisodeResult(t+1, True, info["distance"], trace)

    return EpisodeResult(T, False, info["distance"], trace)
