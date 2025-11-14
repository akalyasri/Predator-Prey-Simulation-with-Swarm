NEAT-Python’s documentation: https://neat-python.readthedocs.io/en/latest/

NEAT Overview: https://neat-python.readthedocs.io/en/latest/neat_overview.html

More detailed description of the algorithm read: Evolving Neural Networks
through Augmenting Topologies by Kenneth O. Stanley (in the Resources folder)

Overview on our code alg:
there is one predator and one prey
- we could do multiple, but right now our simulation uses exactly one of each

They both start at random positions somewhere inside a 100×100 square
- predator brain: goal -> move toward the prey to catch it.
- prey brain: goal -> move away from the predator to avoid being caught.

Right now, the brain can be:
- a dummy heuristic (early tests)
- a neural network evolved by NEAT (training step)

The agents dont see the whole board, each one only gets a small observation vector 
They only know a relative direction, not absolute location
So this is what makes it feel like realistic swarm/local perception

Every time step:
- Each agent's brain outputs (vx, vy) -> like a direction vector
- We multiply that by a speed and we update their position

In one episode:
The simulation runs for up to T = 500 steps

Each step:
- Predator looks at prey -> decides move
- Prey looks at predator -> decides move
- Environment updates positions
- Check: distance < capture_radius?

If YES -> predator wins
If NO after 500 steps -> prey survives

The evolution part:
NEAT is evolving a population of predator brains or prey brains.
we dont hard code how the predator should chase the prey. we let evolution create strategies

The cycle structure for coevolution:
1. Train predators (prey stays fixed)
2. Train prey (predator stays fixed)
3. Add best ones to hall of fame
4. Repeat

This is like an arms race:
- predators evolve to chase better
- prey evolves to escape better