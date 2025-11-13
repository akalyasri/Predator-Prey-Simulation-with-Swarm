# we sample a random genome from a tiny population, wrap it as a controller,
# and run one episode vs a simple greedy prey controller

import os
import numpy as np
import neat

# import environment pieces
from src.env_test.simulation import run_episode, EpisodeResult
from src.env_test.environment import Environment
from src.controllers import make_neat_controller

# simple baseline prey (same logic as test_sim.py)
def prey_escape(obs):
    dx, dy, dist, bias = obs
    step = -np.array([dx, dy], dtype=float)
    step /= (np.linalg.norm(step) + 1e-8)
    return (step * 1.8).tolist()

def load_config():
    cfg_path = os.path.join("config", "neat_config.txt")
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        cfg_path,
    )

def sample_random_genome(config: neat.Config) -> neat.DefaultGenome:
    
    # create a small population and pick one genome (no evolution yet) 
    # -> this guarantees the genome structure matches the config
    
    pop = neat.Population(config)
    # grab any genome from the initialized population
    gid, genome = next(iter(pop.population.items()))
    return genome

def summarize(ep: EpisodeResult):
    print("\nNEAT Integration Episode:")
    print(f"Steps: {ep.steps}")
    print(f"Captured: {'Yes' if ep.captured else 'No'}")
    print(f"Final distance: {ep.final_distance:.2f}")

if __name__ == "__main__":
    # 1. load neat config (num_inputs must match env observe() == 4)
    config = load_config()

    # 2. sample a random genome and wrap it as a predator controller
    genome = sample_random_genome(config)
    pred_ctrl = make_neat_controller(genome, config, speed=2.0)

    # 3. run one episode vs greedy prey
    ep = run_episode(pred_ctrl, prey_escape, env=Environment(), T=500, capture_radius=5.0)
    summarize(ep)

    # 4. output plot
    try:
        from src.env_test.visualize_episode import plot_episode
        plot_episode(ep, save_path="results/neat_integration.png")
    except Exception as e:
        print(f"(Plot skipped) {e}")

# OUTPUT ----------------------------------------------------------------------------
# NEAT Integration Episode:
# Steps: 500
# Captured: No
# Final distance: 100.00
# Saved plot to results/neat_integration.png
# --------------------------------------------------------------------------------------

# the predator never caught the prey (500 = max episode length)
# final distance = 100 -> they ended up on opposite corners
# prey ended at (100, 0) which is the bottom right corner
# predator ended at (0, 0) which is the bottom left corner
# both agents hit the arena boundary and then got stuck there

# this makes sense beacuase NEAT is not evolved yet
# random neural networks are chaotic 
# so this is the expected behavior before evolution
