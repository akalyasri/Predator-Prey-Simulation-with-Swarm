# define separate, simple reward signals for predator vs prey so that
# evolution has a clear optimization target. these are intentionally lightweight
# because NEAT will repeatedly call them thousands of times
import numpy as np

def predator_fitness(ep):
    trace = ep.trace
    if len(trace) < 3:
        return 0.0

    # 1. Capture reward (huge)
    if ep.captured:
        return 1200 + (500 - ep.steps)

    # 2. Distance improvement reward
    dist_reward = 0.0
    for i in range(1, len(trace)):
        prev = trace[i-1]["distance"]
        curr = trace[i]["distance"]
        dist_reward += (prev - curr)

    # 3. Directional reward based on positions
    direction_reward = 0.0
    for i in range(1, len(trace)):
        prev_pred = np.array(trace[i-1]["pred_pos"])
        curr_pred = np.array(trace[i]["pred_pos"])
        prev_prey = np.array(trace[i-1]["prey_pos"])
        curr_prey = np.array(trace[i]["prey_pos"])

        pred_move = curr_pred - prev_pred
        ideal_move = curr_prey - prev_pred  # vector toward prey last step

        if np.linalg.norm(pred_move) > 1e-6 and np.linalg.norm(ideal_move) > 1e-6:
            pred_move /= np.linalg.norm(pred_move)
            ideal_move /= np.linalg.norm(ideal_move)
            direction_reward += pred_move.dot(ideal_move)  # +1 good, -1 bad

    # 4. Smoothness penalty (using positions, not velocities)
    smooth_penalty = 0.0
    for i in range(2, len(trace)):
        v1 = np.array(trace[i-1]["pred_pos"]) - np.array(trace[i-2]["pred_pos"])
        v2 = np.array(trace[i]["pred_pos"])   - np.array(trace[i-1]["pred_pos"])
        smooth_penalty += np.linalg.norm(v2 - v1)

    # weighted combination
    fitness = (
        3.0 * dist_reward +
        1.0 * direction_reward -
        0.4 * smooth_penalty +
        ep.steps
    )

    return float(fitness)



def prey_fitness(ep):
    # reward staying alive
    base = ep.steps

    # reward distance from predator
    dist_bonus = 0.5 * ep.final_distance

    # huge penalty for getting caught
    capture_penalty = -500 if ep.captured else 0

    return base + dist_bonus + capture_penalty



def prey_fitness_against_predator(ep):
    
    trace = ep.trace

    # 1. survive -> strong reward
    fitness = ep.steps * 2.0

    # 2. average distance
    avg_dist = np.mean([step["distance"] for step in trace])
    fitness += avg_dist

    # 3. movement reward (no freezing)
    prey_positions = np.array([step["prey_pos"] for step in trace])
    deltas = np.diff(prey_positions, axis=0)
    total_motion = np.sum(np.linalg.norm(deltas, axis=1))
    fitness += 0.5 * total_motion

    # 4. captured penalty
    if ep.captured:
        fitness -= 800

    return fitness