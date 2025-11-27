import matplotlib.pyplot as plt
import numpy as np

def plot_episode_with_obstacles(ep, env, save_path=None):
    preds = np.array([step["pred_pos"] for step in ep.trace])
    preys = np.array([step["prey_pos"] for step in ep.trace])

    plt.figure(figsize=(7, 7))

    # predator path
    plt.plot(preds[:, 0], preds[:, 1], 'r-', label="Predator Path")
    plt.scatter(preds[0, 0], preds[0, 1], c='darkred', label="Predator Start")
    plt.scatter(preds[-1, 0], preds[-1, 1], c='orange', marker='x', label="Predator End")

    # prey path
    plt.plot(preys[:, 0], preys[:, 1], 'b-', label="Prey Path")
    plt.scatter(preys[0, 0], preys[0, 1], c='navy', label="Prey Start")
    plt.scatter(preys[-1, 0], preys[-1, 1], c='cyan', marker='x', label="Prey End")

    # draw obstacles
    for ob in env.obstacles:
        circle = plt.Circle((ob.x, ob.y), ob.radius, color='gray', alpha=0.4)
        plt.gca().add_patch(circle)

    plt.xlim(0, env.width)
    plt.ylim(0, env.height)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.title(f"Predator vs Evolved Prey (Steps: {ep.steps}, Captured: {ep.captured})")

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
