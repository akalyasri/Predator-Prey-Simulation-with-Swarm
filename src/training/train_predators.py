# first train predators while prey stays fixed
import os
import neat
import pickle
from statistics import mean

from src.core.environment import Environment
from src.core.simulation import run_episode
from src.neat_utils.controllers import make_controller
from src.neat_utils.dummy_controllers import prey_dummy
from src.neat_utils.fitness import predator_fitness

# save results
os.makedirs("results/predator_training", exist_ok=True)

def evaluate_genomes(genomes, config):
    # called by NEAT. assigns fitness to each genome
    for gid, genome in genomes:
        genome.fitness = 0.0

        # build predator controller for this genome
        predator_ctrl = make_controller(genome, config, speed=2.5)

        # run multiple episodes for stability
        episode_fitnesses = []

        for _ in range(5):          # 5 episodes per genome
            env = Environment()     # fresh environment each time

            ep = run_episode(
                predator_ctrl,
                prey_dummy,        # fixed prey
                env,
                T=400              # shorter than 500 to speed training
            )

            episode_fitnesses.append(predator_fitness(ep))

        # use mean fitness across episodes
        genome.fitness = mean(episode_fitnesses)


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

    # run for 30 generations
    winner = pop.run(evaluate_genomes, n=30)

    # save best genome
    with open("results/predator_training/best_predator.pkl", "wb") as f:
        pickle.dump(winner, f)

    print("\nTraining complete. Best genome saved.")

    # save fitness curve
    import matplotlib.pyplot as plt
    gen = range(len(stats.most_fit_genomes))
    best = [g.fitness for g in stats.most_fit_genomes]

    plt.plot(gen, best)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Predator Evolution Progress")
    plt.savefig("results/predator_training/predator_fitness_curve.png")
    plt.close()


if __name__ == "__main__":
    run_training()
