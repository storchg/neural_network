import numpy as np

#src: fundamentals of machine learning slides
def add_bias(data):
    return np.hstack([data, np.ones([data.shape[0], 1])])

# returns a list of weight vectors
def create_weights(list_of_layer_strengths, input_weights, output_weights):
    hiddenlayers = len(list_of_layer_strengths)
    weight_vectors = []
    # input layer
    weight_vector = np.random.rand(input_weights, list_of_layer_strengths[0])
    weight_vectors.append(weight_vector)
    # hidden layers
    for idx, layer_strength in enumerate(list_of_layer_strengths):
        if idx + 1 < hiddenlayers:
            weight_vector = np.random.rand(layer_strength, list_of_layer_strengths[idx+1])
            weight_vectors.append(weight_vector)
        else:
            weight_vector = np.random.rand(layer_strength, output_weights)
            weight_vectors.append(weight_vector)
    return weight_vectors

def shuffle(X, y):
    shuffler = np.array(range(0,len(y)))
    np.random.shuffle(shuffler)
    sample_set = X[shuffler,:]
    labels = y[shuffler]
    return sample_set, labels

