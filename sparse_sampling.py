# Sparse Sampling
import numpy as np
import beetle_simulator
import utils

# inspiration from https://github.com/griffinbholt/decisionmaking-code-py/blob/main/src/ch09.py
GAMMA = 0.9

def generate_simulator():
    seed = 0
    np.random.seed(0)
    forest, beetles = utils.generateRandomForest(10, 2, 200, 5, 5000, seed)
    
    sim = beetle_simulator.Simulator(forest.copy(), beetles.copy())
    return sim


def sparse_sampling(sim, trees, beetles, d, m):
    if d <= 0:
        return (None, 0)
    best_a, best_u = (None, -np.inf)

    actions = sim.get_available_actions()
    for a in actions:
        u = 0.0
        for _ in range(m):
            # rand step
            temp_sim = beetle_simulator.Simulator(trees.copy(), beetles.copy())
            r = temp_sim.take_action(a)
            r += temp_sim.simulate_timestep()
            trees_prime, beetles_prime = temp_sim.trees, temp_sim.beetles

            a_prime, u_prime = sparse_sampling(temp_sim, trees_prime, beetles_prime, d - 1, m)
            u += (r + GAMMA * u_prime) / m

        if u > best_u:
            best_a, best_u = (a,u)
    return best_a, best_u

def main():
    sim = generate_simulator()
    # number of timesteps
    a, u = sparse_sampling(sim, sim.trees, sim.beetles, d = 3, m = 5)
    return a,u

if __name__ == "__main__":
    a,u = main()
    print(a,u)