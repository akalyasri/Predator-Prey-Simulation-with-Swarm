import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from src.core.environment import Environment
from src.core.simulation import run_episode
from src.neat_utils.controllers import make_controller


PRED_PATH = "results/pred_against_prey/best_predator_vs_prey.pkl"
PREY_PATH = "results/prey_training/best_prey.pkl"
CONFIG_PATH = "config/neat_config.txt"

SAVE_DIR = "results/visualizations"
os.makedirs(SAVE_DIR, exist_ok=True)


def load_genome(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_config():
    import neat
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        CONFIG_PATH
    )


def plot_episode_with_obstacles(ep, env, save_path):
    preds = np.array([step["pred_pos"] for step in ep.trace])
    preys = np.array([step["prey_pos"] for step in ep.trace])


    plt.figure(figsize=(8, 7))

    # trajectories
    plt.plot(preds[:, 0], preds[:, 1], '-r', label="Predator Path")
    plt.plot(preys[:, 0], preys[:, 1], '-b', label="Prey Path")

    # start positions
    plt.scatter(preds[0, 0], preds[0, 1], c='darkred', marker='o', label="Predator Start")
    plt.scatter(preys[0, 0], preys[0, 1], c='navy', marker='o', label="Prey Start")

    # end positions
    plt.scatter(preds[-1, 0], preds[-1, 1], c='orange', marker='x', s=80, label="Predator End")
    plt.scatter(preys[-1, 0], preys[-1, 1], c='cyan', marker='x', s=80, label="Prey End")

    # obstacles
    for ob in env.obstacles:
        circle = plt.Circle((ob.x, ob.y), ob.radius,
                            color='gray', alpha=0.3)
        plt.gca().add_patch(circle)

    plt.title(f"Predator vs Evolved Prey (Steps: {ep.steps}, Captured: {ep.captured})")
    plt.xlim(0, env.width)
    plt.ylim(0, env.height)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved visualization to {save_path}")


def main():
    print("\n---- Predator vs Evolved Prey ----")

    config = load_config()

    # Load predator and prey
    predator_genome = load_genome(PRED_PATH)
    prey_genome = load_genome(PREY_PATH)

    predator_ctrl = make_controller(predator_genome, config, speed=2)
    prey_ctrl = make_controller(prey_genome, config, speed=1.3)

    # run simulation
    env = Environment()
    ep = run_episode(predator_ctrl, prey_ctrl, env, T=500)

    # DEBUGGING
    print("\nPrey path deltas (movement each step):")
    prey_path = [step["prey_pos"] for step in ep.trace]

    for i in range(1, len(prey_path)):
        dx = prey_path[i][0] - prey_path[i-1][0]
        dy = prey_path[i][1] - prey_path[i-1][1]
        print(f"Step {i}: dx={dx:.3f}, dy={dy:.3f}")

    print(f"Steps: {ep.steps}")
    print(f"Captured: {ep.captured}")
    print(f"Final distance: {ep.final_distance:.2f}")

    

    plot_episode_with_obstacles(ep, env, os.path.join(SAVE_DIR, "pred_vs_evolved_prey.png"))


if __name__ == "__main__":
    main()
