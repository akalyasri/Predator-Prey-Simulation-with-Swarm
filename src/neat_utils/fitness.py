# define separate, simple reward signals for predator vs prey so that
# evolution has a clear optimization target. these are intentionally lightweight
# because NEAT will repeatedly call them thousands of times
import numpy as np

''''' too complex - as we are retraining pred
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


    # 5. penalty for standing still - should guarentee that the pred doesnt evolve a zero-movement strategy
    stationary_penalty = 0.0
    for i in range(1, len(trace)):
        pred_move = np.linalg.norm(
            np.array(trace[i]["pred_pos"]) - np.array(trace[i-1]["pred_pos"])
        )
        if pred_move < 0.1:
            stationary_penalty += 0.2    # penalize freezing


    # weighted combination
    fitness = (
        3.0 * dist_reward +
        1.0 * direction_reward -
        0.4 * smooth_penalty +
        ep.steps
    )

    return float(fitness)

'''

''' too simple - but okay for first train
def predator_fitness(ep):
    # 1. Big reward for capture
    if ep.captured:
        return 2000 + (500 - ep.steps) * 5

    # 2. Reward getting closer
    start_d = ep.trace[0]["distance"]
    end_d   = ep.trace[-1]["distance"]
    distance_reward = (start_d - end_d) * 5

    # 3. Reward movement (prevents freezing)
    predator_positions = np.array([t["pred_pos"] for t in ep.trace])
    deltas = np.linalg.norm(np.diff(predator_positions, axis=0), axis=1)
    movement_reward = np.sum(deltas) * 2

    return distance_reward + movement_reward

    
'''

def predator_fitness(ep):
    trace = ep.trace

    if len(trace) < 3:
        return 0.0
    
    # 1. big reward for capture
    if ep.captured:
        return 5000 + (500 - ep.steps) * 20   # strong incentive



    # 2. sum of distance reductions each step
    start_d = trace[0]["distance"]
    end_d = trace[-1]["distance"]
    dist_reward = 0

    for i in range(1, len(trace)):
        prev = trace[i-1]["distance"]
        curr = trace[i]["distance"]
        dist_reward += (prev - curr)  # positive if closing in

    
    dist_reward *= 15       # make the distance progress matter



    # 3. movement bonus  - helps prevent freezing
    predator_positions = np.array([t["pred_pos"] for t in trace])
    deltas = np.linalg.norm(np.diff(predator_positions, axis=0), axis=1)
    movement = np.sum(deltas) * 0.2  


    # 4. heading alignment reward 
    pred_pos = predator_positions[-1]
    prev_pred_pos = predator_positions[-2]
    move_vec = pred_pos - prev_pred_pos

    prey_pos = np.array(trace[-1]["prey_pos"])
    vec_to_prey = prey_pos - pred_pos


    if np.linalg.norm(vec_to_prey) > 1e-6 and np.linalg.norm(move_vec) > 1e-6:
        vec_to_prey = vec_to_prey / np.linalg.norm(vec_to_prey)
        move_vec    = move_vec / np.linalg.norm(move_vec)
        heading_reward = 3.0 * np.dot(move_vec, vec_to_prey)
    else:
        heading_reward = 0



    # 5. penalty for increasing distance - prevents running away
    distance_increase = max(0,end_d - start_d)
    runaway_penalty = distance_increase * 2.0   # only negative if predator runs away



    # 6. wall penalty (discourage staying near borders)
    x, y = pred_pos
    margin = 10
    world_size = 100 

    wall_pen = 0

    if x < margin: wall_pen += (margin - x)
    if x > (world_size - margin): wall_pen += (x - (world_size - margin))
    if y < margin: wall_pen += (margin - y)
    if y > (world_size - margin): wall_pen += (y - (world_size - margin))

    wall_pen *= 0.5  

    return dist_reward + movement + heading_reward - runaway_penalty - wall_pen

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