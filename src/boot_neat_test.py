import os
import random
import neat

def evaluate_genomes(genomes, config):

    # for testing - just assigns random fitness values so NEAT can run end to end without real simulation logic 

    for gid, genome in genomes:
        genome.fitness = random.random()  # random fitness in [0, 1)

if __name__ == "__main__":
    # 1. point to your NEAT config file
    config_path = os.path.join("config", "neat_config.txt")

    # 2. load the config
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    # 3. initialize a population using that config
    pop = neat.Population(config)

    # 4. add built in reporters to see what NEAT is doing
    pop.add_reporter(neat.StdOutReporter(True))       # print progress
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    # 5. run the dummy evolution for 3 generations
    winner = pop.run(evaluate_genomes, n=3)

    print("\n Smoke test complete - NEAT ran 3 generations successfully.")
