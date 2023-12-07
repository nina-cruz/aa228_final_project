import numpy as np
import random
import utils


STATE_ALIVE = 1
STATE_BLANK = 0
STATE_FELLED = -1
STATE_EATEN = -2

class Simulator:
    def __init__(self, trees, beetles):
        # n: size of grid (nxn)
        # empty_locations: tuples of indices where no trees
        # beetle_source: tuple of beetle source square
        
        self.trees = trees
        self.beetles = beetles
        self.n = trees.shape[0]

        # Rewards
        self.reward_eaten = -10
        self.reward_cut = -1

        # sample
        self.beetle_repl_rate = 1.1#1.1
        self.beetle_decay_rate = 0.7

        # parameters for Dirichlet beetle movement prob from empty square
        self.stay_empty = 1
        self.empty_to_tree = 4
        self.empty_to_empty = 2

        # parameters for Dirichlet beetle movement prob from tree square
        self.stay_tree = 5
        self.tree_to_tree = 2
        self.tree_to_empty = 1

        self.stay_status = np.array([self.stay_empty, self.stay_tree]) # empty is status 0, alive is status 1
        self.status_to_status = np.array([[self.empty_to_empty, self.empty_to_tree], [self.tree_to_empty, self.tree_to_tree]])

    def replicate(self):
        for i in range(self.n):
            for j in range(self.n):
                if self.trees[(i,j)] > 0:
                    # replicate beetles if tree is alive
                    self.beetles[(i,j)] = np.ceil(self.beetles[(i,j)] * self.beetle_repl_rate)

                else:
                    # decay beetle counts in empty squares
                    self.beetles[(i,j)] = np.floor(self.beetles[(i,j)] * self.beetle_decay_rate)

    def eat(self):
        # eating step -- subtract 1 tree hp for each beetle, hp >= 0 for all trees
        r = 0
        nonzero = np.argwhere(self.beetles)
        for cur in nonzero:
            self.trees[*cur] = max(0, self.trees[*cur] - self.beetles[*cur])
            if self.trees[*cur] == 0:
                self.trees[*cur] = STATE_EATEN
                r += self.reward_eaten
        return r

    def move_beetles(self):
        # create temporary grid to store new beetle counts (need to preserve intial beetle counts to multiply)
        temp_beetles = np.zeros_like(self.beetles)
        directions = np.array([[-1,0],[1,0],[0,-1],[0,1]]) # left, right, above, below
        
        nonzero = np.argwhere(self.beetles)
        for cur in nonzero:
            neighbours = [cur] # Create list of valid neighbours (including the current square)
            for direction in directions: 
                potentialNeighbour = cur + direction
                if self.withinBounds(potentialNeighbour):
                    neighbours.append(potentialNeighbour)        
                    
            originStatus = int(self.trees[*cur] > 0) # 0 if empty, 1 if tree exists
            params = [self.stay_status[originStatus]] # Staying put comes first in params list
            for square in range(1, len(neighbours)):
                neighbour = neighbours[square]
                neighbourStatus = int(self.trees[*neighbour] > 0)
                params.append(self.status_to_status[originStatus, neighbourStatus]) # Set dirichlet parameters for each neighbour

            square_probs = np.random.dirichlet(params)
            
            squareAddBeetles = np.floor(square_probs*self.beetles[*cur]) # Beetles that move to each neighbour

            for square, movedBeetles in zip(neighbours, squareAddBeetles):
                temp_beetles[*square] += movedBeetles
        self.beetles = temp_beetles

    def withinBounds(self, square):
        # Checks that a given square is within the bounds of the grid
        return np.all((0 <= square) & (square < self.n))


    def simulate_timestep(self):
        self.replicate()
        r = self.eat()
        self.move_beetles()
        return r

    def take_action(self,a):
        # 0 = no action
        # 1, ..., n^2 = cut down tree in that square and set beetles to 0
        if np.any(a != np.array([-1,-1])):
            self.beetles[*a] = 0
            self.trees[*a] = STATE_FELLED
            return self.reward_cut # immediate reward
        else:
            return 0

    def get_available_actions(self):
        actions = np.array([-1,-1]) # No action
        living_trees = np.argwhere(self.trees > 0)
        actions = np.vstack((actions, living_trees))
        return actions
    
    def get_available_actions_reduced(self):
        # Actions which involve cutting trees within radius to an eaten tree.
        radius = 3
        actions = np.array([-1,-1], ndmin=2) # No action
        consumed_trees = np.argwhere(self.beetles > 0)
        living_trees = np.argwhere(self.trees > 0)
        for living_tree in living_trees:
            for consumed_tree in consumed_trees:
                if np.linalg.norm(living_tree - consumed_tree) < radius:
                    actions = np.append(actions,np.expand_dims(living_tree, axis=0), axis=0)
                    break
        return actions
        
            
def main():
    seed = 0
    np.random.seed(0)
    forest, beetles = utils.generateRandomForest(10, 2, 200, 5, 5000, seed)
    
    sim = Simulator(forest.copy(), beetles.copy())
    for i in range(50):
        sim.simulate_timestep()
        

    
    sim.take_action(np.array([-1,-1]))

    sim.get_available_actions_reduced()


    utils.plotTrees(sim.trees, max_health=1000)


if __name__ == "__main__":
    main()