from math import exp
from operator import indexOf
import numpy as np
from neural_network.helpers.encoding import one_hot_encoding
from neural_network.helpers.manipulations import add_bias, create_weights, shuffle
from neural_network.helpers.cost import Cost
from neural_network.helpers.activation import Activation

class NN:
    def __init__(self, activation:Activation=Activation(), output_activation: Activation=Activation("softmax"), cost: Cost=Cost(), learning_rate = 0.1):
        self.errors = []
        self.weights = []
        self.activation = activation
        self.output_activation = output_activation
        self.cost = cost
        self.learning_rate = learning_rate


    def fit(self, X, y, max_iters=1, hidden_layers = [2, 3, 4]):
        original_labels, y = one_hot_encoding(y) 
        X = add_bias(X) 
        self.weights = create_weights(hidden_layers, X.shape[1], len(original_labels))  # weights is a list of numpy arrays.
        # per epoch
        for epoch in range(max_iters):
            X, y = shuffle(X,y)
            # per sample
            for idx, sample in enumerate(X):
                output = self.feed_sample_forward(sample)
                #print("err-Grundlagen: ", y[idx,:], output, y[idx,:]-output)
                self.backpropagate((y[idx, :]-output), output)
                self.errors.append(self.cost.calc(output, y))
        return self.errors
    
    def backpropagate(self, sample_error, sample_output):
        gradients = []
        # get gradients
        current_gradient = None
        self.weights.reverse() # von hinten nach vorne!
        for layer_idx in range(0,len(self.weights)):
            if layer_idx == 0: # output-weights
                print(sample_output[:, None], sample_error[:, None].T)
                current_gradient = np.matmul(sample_output[:, None],sample_error[:, None].T)
                gradients.append(current_gradient)
                print(f"output shape: {current_gradient.shape}")
            else:
                current_gradient = np.matmul(self.weights[layer_idx-1], current_gradient)
                for i, e in enumerate(current_gradient):
                    current_gradient[i] = self.activation.dsigmoid(e)
                gradients.append(current_gradient)
                #print("current gradient: ", current_gradient)
                
        #gradients.reverse()
        print("weight shapes: ", [w.shape for w in self.weights])
        print("gradient shapes: ", [w.shape for w in gradients])
        #print("gradients:\n",gradients, "\n")
        
        # correct weights
        for idx in range(0, len(self.weights)):
            #print(self.weights[idx], self.learning_rate, gradients[idx])
            print(f"weights {idx}:\n", self.weights[idx], f"\ngradients {idx}:\n",gradients[idx], "\n")
            self.weights[idx] += (self.learning_rate * current_gradient[idx])
        #    print(f"weights {idx}:\n", self.weights[idx], f"\ngradients {idx}:\n",gradients[idx], "\n")
        self.weights.reverse()

    def feed_sample_forward(self, sample):
            for idx, layer in enumerate(self.weights):
                if idx == len(self.weights)-1:  
                    return self.feedforward(sample, layer, is_output=True)
                else:
                    sample = self.feedforward(sample, layer)

    def feedforward(self, sample, nodes, is_output:bool=False):
        # calculate feed forward values
        forward_values = nodes
        ## TODO: Warum darf ich in "is_output" die gewichte nicht mehr anwenden? Feature ist dann irgendwie immer 0.
        if not is_output:
            for idx, feature in enumerate(sample):
                forward_values[idx] *= feature
    
        # calculate sum of forward_values
        next_layer_depth = nodes[0].shape[0]
        activations = np.zeros(next_layer_depth)
        for node in forward_values:
            activations += node

        # into activation function.
        if is_output:
            activations = self.output_activation.calc(activations)
            #print("activations", activations)
        else:
            for idx, activation_value in enumerate(activations):
                activations[idx] = self.activation.calc(activation_value)

        return activations

