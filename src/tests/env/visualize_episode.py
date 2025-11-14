import matplotlib.pyplot as plt
import numpy as np

def plot_episode(ep, env=None, save_path=None):
    # plot predator vs prey trajectories from an episode 
    preds = np.array([p for p, _ in ep.trace])
    preys = np.array([q for _, q in ep.trace])

    plt.figure(figsize=(7, 6))
    plt.plot(preds[:, 0], preds[:, 1], '-r', label="Predator Path")
    plt.plot(preys[:, 0], preys[:, 1], '-b', label="Prey Path")

    # start and end points
    plt.scatter(preds[0, 0], preds[0, 1], c='darkred', marker='o', label="Predator Start")
    plt.scatter(preys[0, 0], preys[0, 1], c='navy', marker='o', label="Prey Start")
    plt.scatter(preds[-1, 0], preds[-1, 1], c='orange', marker='x', label="Predator End")
    plt.scatter(preys[-1, 0], preys[-1, 1], c='cyan', marker='x', label="Prey End")


    if env is not None and hasattr(env, "obstacles"):
        for (ox, oy, w, h) in env.obstacles:
            rect = plt.Rectangle((ox, oy), w, h,
                                 color='gray', alpha=0.4, label=None)
            plt.gca().add_patch(rect)

        # add legend entry only once
        plt.plot([], [], color='gray', alpha=0.4, label="Obstacle")

    # axes settings
    if env is not None:
        plt.xlim(0, env.width)
        plt.ylim(0, env.height)
    else:
        plt.xlim(0, 100)
        plt.ylim(0, 100)

    plt.gca().set_aspect('equal', adjustable='box')


    plt.title(f"Predator-Prey Trajectory (Steps: {ep.steps}, Captured: {ep.captured})")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    #plt.xlim(0, 100)
    #plt.ylim(0, 100)
    #plt.autoscale() # autoscale axes to fit all points
    #plt.gca().set_aspect('equal', adjustable='box') # fixes distortion

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
    else:
        plt.show()