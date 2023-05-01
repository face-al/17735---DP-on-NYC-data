import diffprivlib.mechanisms 
import numpy as np

def gaussian(train, epsilon):
    mechanism = diffprivlib.mechanisms.Gaussian(epsilon=epsilon, delta=0.5, sensitivity=0.5)
    result = np.copy(train)
    for x in range(train.shape[0]):
        for y in range(train.shape[1]):
            result[x][y] = mechanism.randomise(result[x][y])
    return result

def laplace(train, epsilon):
    mechanism = diffprivlib.mechanisms.Laplace(epsilon=epsilon, sensitivity=0.5)
    result = np.copy(train)
    for x in range(train.shape[0]):
        for y in range(train.shape[1]):
            result[x][y] = mechanism.randomise(result[x][y])
    return result

