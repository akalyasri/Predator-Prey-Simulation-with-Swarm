# train prey while predator is fixed (using the evolved predator)
import os
import neat
import pickle
from statistics import mean

from src.core.environment import Environment
from src.core.simulation import run_episode
from src.neat_utils.controllers import make_controller
from src.neat_utils.fitness import prey_fitness_against_predator


PREDATOR_MODEL_PATH = "results/predator_training/best_predator.pkl"

os.makedirs("results/prey_training", exist_ok=True)


def load_predator(config):
    with open(PREDATOR_MODEL_PATH, "rb") as f:
        predator_genome = pickle.load(f)
    predator_ctrl = make_controller(predator_genome, config, speed=2.0)
    return predator_ctrl



# evaluate prey genomes
def evaluate_genomes(genomes, config):

    # load predator ONCE outside the loop -> efficiency
    predator_ctrl = load_predator(config)

    for gid, genome in genomes:
        genome.fitness = 0.0

        prey_ctrl = make_controller(genome, config, speed=1.3)
        episode_scores = []

        for _ in range(5):   # evaluate each prey 5 times for stability
            env = Environment()
            ep = run_episode(predator_ctrl, prey_ctrl, env, T=500)
            score = prey_fitness_against_predator(ep)
            episode_scores.append(score)

        genome.fitness = mean(episode_scores)



def run_training():
    config_path = os.path.join("config", "neat_config.txt")

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    winner = pop.run(evaluate_genomes, n=30)

    with open("results/prey_training/best_prey.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\nPrey training complete. Best prey saved.")

    # save fitness curve
    import matplotlib.pyplot as plt
    gen = range(len(stats.most_fit_genomes))
    best = [g.fitness for g in stats.most_fit_genomes]

    plt.plot(gen, best)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Prey Evolution Progress")
    plt.savefig("results/prey_training/prey_fitness_curve.png")
    plt.close()


if __name__ == "__main__":
    run_training()
