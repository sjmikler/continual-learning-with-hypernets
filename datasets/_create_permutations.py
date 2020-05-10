import numpy as np

if __name__ == '__main__':
    permutations = np.random.rand(100, 784)
    permutations = permutations.argsort(1)
    permutations[0] = np.arange(784)
    
    np.save('_permutations', permutations)
    