import numpy as np
import random

class Simulator:
    def __init__(self, n, empty_locations, beetle_source):
        # n: size of grid (nxn)
        # empty_locations: tuples of indices where no trees
        # beetle_source: tuple of beetle source square
        self.trees = np.ones((n,n))*5000
        for loc in empty_locations:
            self.trees[loc] = 0
        self.source = beetle_source
        self.beetles = np.zeros((n,n))
        self.beetles[beetle_source] = 200
        self.n = n

        # parameters
        self.beetle_repl_rate = 1.1
        self.beetle_decay_rate = 0.7

        # parameters for Dirichlet beetle movement prob from empty square
        self.stay_empty = 1
        self.empty_to_tree = 4
        self.empty_to_empty = 2

        # parameters for Dirichlet beetle movement prob from tree square
        self.stay_tree = 5
        self.tree_to_tree = 2
        self.tree_to_empty = 1

    def replicate_and_eat(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.trees[(i,j)] > 0:
                    # replicate beetles if tree is alive
                    self.beetles[(i,j)] = np.ceil(self.beetles[(i,j)] * self.beetle_repl_rate)

                    # eating step -- subtract 1 tree hp for each beetle, hp >= 0 for all trees
                    self.trees[(i,j)] = max(0, self.trees[(i,j)] - self.beetles[(i,j)])
                else:
                    # decay beetle counts in empty squares
                    self.beetles[(i,j)] = np.floor(self.beetles[(i,j)] * self.beetle_decay_rate)
    
    def move_beetles(self):
        # create temporary grid to store new beetle counts (need to preserve intial beetle counts to multiply)
        temp_beetles = np.zeros_like(self.beetles)

        for i in range(self.n):
            for j in range(self.n):
                neighbors = [] # neighbors on the grid
                params = [] # dirichlet parameters

                # moving from a tree square
                if self.trees[(i,j)] > 0:
                    # left
                    if j-1 >= 0:
                        dir = self.tree_to_tree if self.trees[(i,j-1)] > 0 else self.tree_to_empty
                        neighbors.append((i,j-1))
                        params.append(dir)
                    # right
                    if j+1 < self.n:
                        dir = self.tree_to_tree if self.trees[(i,j+1)] > 0 else self.tree_to_empty
                        neighbors.append((i,j+1))
                        params.append(dir)
                    # above
                    if i-1 >= 0:
                        dir = self.tree_to_tree if self.trees[(i-1,j)] > 0 else self.tree_to_empty
                        neighbors.append((i-1,j))
                        params.append(dir)
                    # below
                    if i+1 < self.n:
                        dir = self.tree_to_tree if self.trees[(i+1,j)] > 0 else self.tree_to_empty
                        neighbors.append((i+1,j))
                        params.append(dir)
                    # add stay in square (i,j)
                    neighbors.append((i,j))
                    params.append(self.stay_tree)

                # moving from an empty square
                else:
                    # left
                    if j-1 >= 0:
                        dir = self.empty_to_tree if self.trees[(i,j-1)] > 0 else self.empty_to_empty
                        neighbors.append((i,j-1))
                        params.append(dir)
                    # right
                    if j+1 < self.n:
                        dir = self.empty_to_tree if self.trees[(i,j+1)] > 0 else self.empty_to_empty
                        neighbors.append((i,j+1))
                        params.append(dir)
                    # above
                    if i-1 >= 0:
                        dir = self.empty_to_tree if self.trees[(i-1,j)] > 0 else self.empty_to_empty
                        neighbors.append((i-1,j))
                        params.append(dir)
                    # below
                    if i+1 < self.n:
                        dir = self.empty_to_tree if self.trees[(i+1,j)] > 0 else self.empty_to_empty
                        neighbors.append((i+1,j))
                        params.append(dir)
                    # add stay in square (i,j)
                    neighbors.append((i,j))
                    params.append(self.stay_empty)

                square_probs = np.random.dirichlet(params)
                # move beetles with proportions from square_probs
                for ind,square in enumerate(neighbors):
                    temp_beetles[square] += np.floor(square_probs[ind]*self.beetles[(i,j)])
        
        self.beetles = temp_beetles

    def simulate_timestep(self):
        self.replicate_and_eat()
        self.move_beetles()


def main():
    sim = Simulator(10,[(0,5),(0,2),(2,6),(2,7),(3,2),(4,9),(3,9),(8,7)],(0,0))

    for i in range(10):
        sim.simulate_timestep()
    print(sim.beetles)
    print(sim.trees)

if __name__ == "__main__":
    main()