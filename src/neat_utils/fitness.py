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
    
    # higher fitness for surviving more time and ending far from predator
    

    survival_bonus = ep.steps             # max 500
    distance_bonus = ep.final_distance    # typically 0–100

    # if prey survives entire episode -> huge reward
    if not ep.captured:
        return survival_bonus * 2 + distance_bonus

    # if captured early -> punish
    return survival_bonus * 0.5 + distance_bonus
