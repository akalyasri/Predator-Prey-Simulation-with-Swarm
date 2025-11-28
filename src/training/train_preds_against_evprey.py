# third training stage: evolve predators against evolved prey
# this completes the basic competitive coevolution loop

import os
import neat
import pickle
from statistics import mean

from src.core.environment import Environment
from src.core.simulation import run_episode
from src.neat_utils.controllers import make_controller
from src.neat_utils.fitness import predator_fitness

# load the already evolved prey brain
PREY_MODEL_PATH = "results/prey_training/best_prey.pkl"

# save results
SAVE_DIR = "results/pred_against_prey"
os.makedirs(SAVE_DIR, exist_ok=True)


# load evolved prey controller
def load_evolved_prey(config):
    with open(PREY_MODEL_PATH, "rb") as f:
        prey_genome = pickle.load(f)

    prey_controller = make_controller(prey_genome, config, speed=1.3)
    return prey_controller


# fitness evaluation for predators
def evaluate_genomes(genomes, config):
    prey_controller = load_evolved_prey(config)

    for gid, genome in genomes:
        genome.fitness = 0.0

        # build predator controller
        predator_ctrl = make_controller(genome, config, speed=2)

        fitness_scores = []

        # eval on multiple environments for robustness
        for _ in range(5):
            env = Environment()

            ep = run_episode(
                predator_ctrl,
                prey_controller,
                env,
                T=500
            )

            fitness_scores.append(predator_fitness(ep))

        genome.fitness = mean(fitness_scores)


# run training
def run_training():
    config_path = os.path.join("config", "neat_config.txt")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    print("\n--- Training predators against evolved prey ---\n")

    # run for 30 generations
    winner = pop.run(evaluate_genomes, n=30)

    # save best predator
    with open(f"{SAVE_DIR}/best_predator_vs_prey.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\nTraining complete. Saved new best predator.\n")

    # plot fitness curve
    import matplotlib.pyplot as plt
    gens = range(len(stats.most_fit_genomes))
    best = [g.fitness for g in stats.most_fit_genomes]

    plt.plot(gens, best)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Predator Evolution Against Evolved Prey")
    plt.savefig(f"{SAVE_DIR}/fitness_curve.png")
    plt.close()


if __name__ == "__main__":
    run_training()
