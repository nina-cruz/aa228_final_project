import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np



def generateRandomForest(nLength, nSources, nBeetles, nEmpty, maxHealth, seed = None):
    '''Method for generating a forest on which to run the simulator'''
    rng = np.random.default_rng(seed)
    trees = np.ones((nLength, nLength))*maxHealth
    beetles = np.zeros((nLength, nLength))


    # The lines below select the prescribed number of empty and source squares and set them accordingly
    emptySquares = np.unravel_index(rng.choice(trees.size, replace=False, size=nEmpty), trees.shape)
    trees[emptySquares] = 0

    sourceSquares = np.unravel_index(rng.choice(beetles.size, replace=False, size=nSources), beetles.shape)
    beetles[sourceSquares] = nBeetles
    
    return trees, beetles


def plotTrees(trees):
    trees = ((trees / 5000) * 100).astype(int)
    plt.figure(figsize=trees.shape)
    plt.imshow(trees, cmap=mpl.colormaps['viridis'])
    plt.show()

def plotBeetles(beetles):
    pass


def main():
    forest, beetles = generateRandomForest(10, 2, 100, 5, 5000)
    plotTrees(forest)


if __name__ == "__main__":
    main()

