import numpy as np


def load_data(quantity_each=1000, split_ratio=0.8):
    paths = {}
    paths["axe"] = './data/npy/axe.npy'
    paths["bat"] = './data/npy/bat.npy'

    dataset = {}

    X = None
    Y = None
    total = 0

    for label in paths:
        path = paths[label]
        dataset[label] = np.load(path)[:quantity_each]

        if X is None:
            X = dataset[label]
            Y = np.array([0 for i in range(quantity_each)])
        else:
            X = np.append(X, dataset[label], axis=0)
            Y = np.append(Y, np.array([int(total / quantity_each) for i in range(quantity_each)]))

        total += quantity_each
        
    Y = Y.reshape(total, 1)

    # Shuffling the arrays
    np.random.seed(1)
    np.random.shuffle(X)

    np.random.seed(1)
    np.random.shuffle(Y)

    train_size = int(split_ratio * total)

    X_train, X_test = np.split(X, [train_size])
    Y_train, Y_test = np.split(Y, [train_size])

    return X_train.T, Y_train.T, X_test.T, Y_test.T



def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))

    return A

def relu(Z):
    A = np.maximum(0,Z)
        
    return A

acitvation_functions = {
    "sigmoid": sigmoid,
    "relu": relu
}

def sigmoid_backward(dA, Z):
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    return dZ

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    return dZ

backward_activation_functions = {
    "sigmoid": sigmoid_backward,
    "relu": relu_backward
}


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 lines of code)
    cost = -np.sum( Y * np.log(AL) + (1 - Y) * np.log(1 - AL) ) / m
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost
