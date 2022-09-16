
from neural_network.helpers.activation import Activation
from array import array
import numpy as np


class Perceptron:
    ALLOWED_TYPES = ["dualout", "multiout", "hidden"]
    def __init__(self, max_iterations=10, activation_function: Activation=Activation("step"), type: str = "perc"):
        self.max_iterations = max_iterations
        self.activation = activation_function
        self.type = type

    def train(self, sample_set: array, labels: array, shuffle = False):
        if len(np.unique(labels)) > 2 and self.type == "dualout":
            raise("Perceptron is a linear classifier, it can only differentiate two classes.")
        bias = 1
        self.weights = np.zeros([bias + sample_set.shape[1]])
        weigths = self.weights
        # we shuffle just in case the 
        for iteration in range(0, self.max_iterations):
            
            if shuffle:
                shuffler = np.array(range(0,len(sample_set)))
                np.random.shuffle(shuffler)
                sample_set = sample_set[shuffler,:]
                labels = labels[shuffler]
            for sample_index, sample in enumerate(sample_set):
                sample = np.concatenate([[1], sample]) # add bias
                output_for_sample = self.activation.calc(np.dot(weigths, sample)) # step function
                for index in range(0, weigths.shape[0]):
                    weigths[index] = weigths[index] + (labels[sample_index]-output_for_sample)*sample[index]
        self.weights = weigths

    def predict(self, sample_set: array):
        output = []
        for sample in sample_set:
            sample = np.concatenate([[1], sample]) # add bias
            output.append(self.activation.calc(np.dot(self.weights, sample)))
        return output

