# test for verifying predator/prey fitness works

from src.core.environment import Environment
from src.core.simulation import run_episode
from src.neat_utils.dummy_controllers import pred_dummy, prey_dummy
from src.neat_utils.fitness import predator_fitness, prey_fitness

# print("Initializing environment") - DEBUGGING
env = Environment()

# print("Running episode with dummy controllers") - DEBUGGING
ep = run_episode(pred_dummy, prey_dummy, env, T=200)

print("\nEpisode Result:")
print(f"Steps:          {ep.steps}")
print(f"Captured:       {ep.captured}")
print(f"Final Distance: {ep.final_distance:.2f}")

print("\nFitness Scores:")
print(f"Predator Fitness: {predator_fitness(ep):.2f}")
print(f"Prey Fitness:     {prey_fitness(ep):.2f}")


# OUTPUT ---------------------------------------------------
# Episode Result:
# Steps:          38
# Captured:       True
# Final Distance: 4.81

# Fitness Scores:
# Predator Fitness: 1462.00
# Prey Fitness:     -459.59
# ----------------------------------------------------------

# the predator caught the prey in 38 steps 
# -> a fast capture -- so predator fitness should be high

# since capture happened, the predator gets a capture bonus + time reward
# the prey gets a capture penalty

# 1000 + (500 - 38) = 1000 + 462 = 1462
# so the predator fitness is correct

# prey fitness = -459.59
# base = ep.steps # 38
# dist_bonus = 0.5 * ep.final_distance  # 0.5 * 4.81 ≈ 2.4
# capture_penalty = -500
# fitness = 38 + 2.4 - 500 = -459.6
# prey survived 38 steps but it got caught 
# -> receives heavy penalty, result is a negative score -> correct

# this is how prey evolution knows it shouldnt get caught
