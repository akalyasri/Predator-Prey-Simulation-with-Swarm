# for testing 
import numpy as np
from .simulation import run_episode

# predator moves toward the prey based on relative position
def predator_greedy(obs):
    dx, dy, dist, bias = obs
    step = np.array([dx, dy])
    step = step / (np.linalg.norm(step) + 1e-8)
    return step * 2.0  # speed

# prey always moves directly away from predator
def prey_escape(obs):
    dx, dy, dist, bias = obs
    step = -np.array([dx, dy])
    step = step / (np.linalg.norm(step) + 1e-8)
    return step * 1.8   #fixed speed


def summarize_episode(ep):
    
    print("\nEpisode Summary:")
    print(f"Steps taken: {ep.steps}")
    print(f"Captured: {'Yes' if ep.captured else 'No'}")
    print(f"Final distance: {ep.final_distance:.2f}")
    print(f"Trace length: {len(ep.trace)}")

    # print start and end positions (rounded)
    start_pred, start_prey = ep.trace[0]
    end_pred, end_prey = ep.trace[-1]
    print(f"Start (Predator): {[round(x,2) for x in start_pred]}")
    print(f"Start (Prey): {[round(x,2) for x in start_prey]}")
    print(f"End (Predator): {[round(x,2) for x in end_pred]}")
    print(f"End (Prey): {[round(x,2) for x in end_prey]}")

    # for debugging - preview a few positions
    if len(ep.trace) > 5:
        print("\nSample trajectory points:")
        for i in np.linspace(0, len(ep.trace) - 1, 5, dtype=int):
            pred, prey = ep.trace[i]
            print(f" Step {i:<3}: Predator {np.round(pred,1)} | Prey {np.round(prey,1)}")


if __name__ == "__main__":
    ep = run_episode(predator_greedy, prey_escape)
    summarize_episode(ep)

    # plotting 
    from visualize_episode import plot_episode
    plot_episode(ep, save_path="results/sample_trajectory.png")