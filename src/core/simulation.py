# runs one episode of predatory and prey interaction
from dataclasses import dataclass
from .environment import Environment

@dataclass
class EpisodeResult:
    steps: int
    captured: bool
    final_distance: float
    trace: list         # stores all positions - for visualization later

def run_episode(pred_controller, prey_controller,
                env=None, T=500, capture_radius=5.0):
    
    if env is None:
        env = Environment(capture_radius=capture_radius)
    
    env.reset()
    trace = []

    for t in range(T):

        obs_pred, obs_prey = env.observe()

        # each controller outputs velocity along (x, y)
        a_pred = pred_controller(obs_pred)
        a_prey = prey_controller(obs_prey)

        _, info = env.step(a_pred, a_prey)
        trace.append(info["positions"])
        
        if info["captured"]:
            return EpisodeResult(t+1, True, info["distance"], trace)
    
    # if prey survives entire epsiode
    return EpisodeResult(T, False, info["distance"], trace)
