# Sparse Sampling
import numpy as np
import beetle_simulator_unfixed as beetle_sim
import utils
from multiprocessing import pool
import time

# inspiration from https://github.com/griffinbholt/decisionmaking-code-py/blob/main/src/ch09.py
GAMMA = 0.9

def generate_simulator():
    seed = 0
    np.random.seed(0)
    forest, beetles = utils.generateRandomForest(10, 1, 200, 5, 500, seed)
    
    sim = beetle_sim.Simulator(forest.copy(), beetles.copy())
    return sim

def repeat_action(sim, trees, beetles, d, m, a):
    # if d <= 0: # Rollout
    #     return (None, 0)
    # u = 0.0
    # for _ in range(m):
    #     # rand step
    #     temp_sim = beetle_sim.Simulator(trees.copy(), beetles.copy())
    #     r = temp_sim.take_action(a)
    #     r += temp_sim.simulate_timestep()
    #     trees_prime, beetles_prime = temp_sim.trees, temp_sim.beetles

    #     a_prime, u_prime = repeat_action(temp_sim, trees_prime, beetles_prime, d - 1, m, a)
    #     u += (r + GAMMA * u_prime) / m
    # return a, u
    r = 0
    for _ in range(d):
        r += GAMMA**d * sim.take_action(a)
        r += GAMMA**d * sim.simulate_timestep()
    return [-1,-1], r


def sparse_sampling(sim, trees, beetles, d, m):
    if d <= 0: # Rollout
        aroll = [-1,-1] # What action to try
        droll = 10 # For how long
        mroll = 1 # How many sampled states for each action
        aroll, uroll = repeat_action(sim, trees, beetles, droll, mroll, aroll)
        return (aroll, uroll)

    best_a, best_u = (None, -np.inf)
    actions = sim.get_available_actions_reduced()
    for a in actions:
        u = 0.0
        for _ in range(m):
            # rand step
            temp_sim = beetle_sim.Simulator(trees.copy(), beetles.copy())
            r = temp_sim.take_action(a)
            r += temp_sim.simulate_timestep()
            trees_prime, beetles_prime = temp_sim.trees, temp_sim.beetles
            a_prime, u_prime = sparse_sampling(temp_sim, trees_prime, beetles_prime, d - 1, m)
            u += (r + GAMMA * u_prime) / m
        if u > best_u:
            best_a, best_u = (a,u)
    return best_a, best_u

def main(nsteps):
    sim = generate_simulator()
    timesteps = nsteps
    actions = []
    utility = 0

    for i in range(timesteps):
        start = time.time()
        a, u = sparse_sampling(sim, sim.trees, sim.beetles, d = 2, m = 5)
        actions.append(a)
        print("Action taken: {}".format(a))
        utility += sim.take_action(a)
        utility += sim.simulate_timestep()
        end = time.time()
        utils.plotTrees(sim.trees, i, max_health = 500)
        print("Time elapsed for timestep {} of {}: {}".format(i+1, timesteps, end-start))
    return actions, utility

def random_simulation(nsteps):
    sim = generate_simulator()
    timesteps = nsteps
    u = 0
    for i in range(timesteps):
        available_actions = sim.get_available_actions_reduced()
        a = available_actions[np.random.choice(len(available_actions))]
        u += sim.take_action(a)
        u += sim.simulate_timestep()
    return u, sim.trees

def noaction_simulation(nsteps):
    sim = generate_simulator()
    timesteps = nsteps
    u = 0
    a = [-1,-1]
    for i in range(timesteps):
        u += sim.take_action(a)
        u += sim.simulate_timestep()
    return u, sim.trees


if __name__ == "__main__":
    timesteps = 50
    u_random, random_trees = random_simulation(timesteps)
    print("Utility random: ", u_random)
    utils.plotTrees(random_trees, "RandomAction", 500)
    u_none, none_trees = noaction_simulation(timesteps)
    utils.plotTrees(none_trees, "NoAction", 500)
    print("Utility of no action: ", u_none)
    a,u = main(timesteps)
    # print(a,u)
    print("Utility: ", u)
