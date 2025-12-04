import os
import pickle
import neat
import numpy as np

from src.core.environment import Environment
from src.core.simulation import run_episode
from src.neat_utils.controllers import make_controller
from src.visualizations.viz_utils import plot_episode_with_obstacles
from src.neat_utils.dummy_controllers import GreedyPreyDummy
from src.visualizations.animate_episode import animate_episode



PRED_PATH = "results/predator_training/best_predator.pkl"
PREY_PATH = "results/prey_training/best_prey.pkl"
CONFIG_PATH = "config/neat_config.txt"


def load_network(path, config):
    # load genome and create NEAT controller

    with open(path, "rb") as f:
        genome = pickle.load(f)

    return make_controller(genome, config, speed=2.0)


def main():
    os.makedirs("results/visualizations", exist_ok=True)

    # load config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )

    # new environment with obstacles
    env = Environment(num_obstacles=3)

    # load trained predator and prey
    pred_ctrl = load_network(PRED_PATH, config)

    prey_ctrl = load_network(PREY_PATH, config)
    #prey_ctrl = GreedyPreyDummy(env)


    

    # run a single visualization episode
    ep = run_episode(pred_ctrl, prey_ctrl, env, T=500)

    print("\nDEBUG TRACE ENTRY:")
    print(ep.trace[0])
    print(ep.trace[1])

    print("\n---- Predator vs Evolved Prey ----")
    print(f"Steps: {ep.steps}")
    print(f"Captured: {ep.captured}")
    print(f"Final distance: {ep.final_distance:.2f}")

    # save plot
    out_path = "results/visualizations/pred_vs_prey.png"
    plot_episode_with_obstacles(ep, env, save_path=out_path)
    print(f"Saved visualization to {out_path}")

    # save mp4 animation
    animate_episode(ep, env, save_path="results/visualizations/chase4.mp4")



if __name__ == "__main__":
    main()
