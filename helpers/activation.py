from math import exp
import numpy as np

# This class defines the activation function. The type of function is given as str. In Version 1.0 supports "relu", "softmax", "step" and "sig"

class Activation:
    VERSION = 1.0
    ALLOWED_TYPES = ["relu", "sig", "softmax", "step"]
    def __init__(self, type: str = "relu"):
        if type not in Activation.ALLOWED_TYPES:
            raise(f"The provided type of activation function is currently not  supported. Try one of {Activation.ALLOWED_TYPES}")
        self.type = type
    
    def calc(self, input):
        if self.type == "relu":
            return self.relu(input)
        if self.type == "sig":
            return self.sigmoid(input)
        if self.type == "softmax":
            return self.softmax(input)
        if self.type == "step":
            return self.step(input)
    
    #src: fundamentals of machine learning slides
    def relu(self, value):
        return max(0, value)
    
    #src: fundamentals of machine learning slides
    #TODO: value vs values? 
    def drelu(self, values):
        res = np.zeros(values.shape)
        res[values > 0] = 1
        return res

    # logistic function, src: wikipedia 09-11-2022
    def sigmoid(self, value):
        return (1.0 / (1 + exp(-value)))

    # src: fundamentals of machine learning slides
    def dsigmoid(self, value):
        for i,val in enumerate(value):
            value[i] = exp(-val)/(1 + exp(-val)) ** 2 
        return value

    # softmax for multiclass output layer, src: fundamentals of machine learning slides
    def softmax(self, values):
        return np.exp(values)/np.sum(np.exp(values))

    
    def step(self, value):
        return 1 if value > 0 else 0