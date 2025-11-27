# define separate, simple reward signals for predator vs prey so that
# evolution has a clear optimization target. these are intentionally lightweight
# because NEAT will repeatedly call them thousands of times

def predator_fitness(ep):
    # strong incentive for capture
    if ep.captured:
        # 1000 + faster is better
        #return 1000 + (500 - ep.steps)
        return 500 + (300 - ep.steps)
    
    # otherwise reward closeness
    return 1.0 / (1e-6 + ep.final_distance)


def prey_fitness(ep):
    # reward staying alive
    base = ep.steps

    # reward distance from predator
    dist_bonus = 0.5 * ep.final_distance

    # huge penalty for getting caught
    capture_penalty = -500 if ep.captured else 0

    return base + dist_bonus + capture_penalty

def prey_fitness_against_predator(ep):
    
    import numpy as np

    # 1. survive as long as possible
    fitness = ep.steps * 2.0   # up to 1000

    # 2. average distance during the whole episode
    dists = [np.linalg.norm(pred - prey) for pred, prey in ep.trace]
    avg_distance = np.mean(dists)
    fitness += avg_distance

    # 3. reward movement -> prevents freezing
    prey_positions = np.array([prey for _, prey in ep.trace])
    deltas = np.diff(prey_positions, axis=0)
    step_magnitudes = np.linalg.norm(deltas, axis=1)
    total_motion = np.sum(step_magnitudes)
    fitness += 0.5 * total_motion

    # 4. penalty if caught
    if ep.captured:
        fitness -= 800

    return fitness