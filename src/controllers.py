# wrap a NEAT genome as a movement controller for our agents
from typing import Callable, Sequence, Tuple
import numpy as np
import neat

def _preprocess_obs(obs: Sequence[float]) -> np.ndarray:
   
    # right now our obs are already scaled reasonably:
    # dx, dy in [-(world), +(world)] but direction gets normalized,
    # distance >= 0, bias = 1
    
    return np.asarray(obs, dtype=np.float32)

def make_controller(
    genome: neat.DefaultGenome,
    config: neat.Config,
    speed: float = 2.0,
) -> Callable[[Sequence[float]], Tuple[float, float]]:
    # turn a NEAT genome into a function(obs)->(vx, vy)
    # assumes config.num_outputs == 2 and num_inputs matches env observations
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    def controller(obs: Sequence[float]) -> Tuple[float, float]:
        v = _preprocess_obs(obs)
        out = net.activate(v)          # length = 2
        # most activations are tanh/sigmoid; ensure safe bounds anyway
        vx = float(np.clip(out[0], -1.0, 1.0)) * speed
        vy = float(np.clip(out[1], -1.0, 1.0)) * speed
        return vx, vy

    return controller