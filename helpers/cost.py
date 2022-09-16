import numpy as np

## here we define cost functions
#TODO: extend supported types.

class Cost:
    VERSION = 1.0
    SUPPORTED_TYPES = ["crossentro"]
    def __init__(self, type: str="crossentro"):
        if type not in Cost.SUPPORTED_TYPES:
            raise(f"Unsupported type error. Try to pass one of these as your cost function type: {Cost.SUPPORTED_TYPES}")
        self.type = type
    
    #TODO: implement cost functions
    def calc(self, prediction, labels):
        if self.type == "crossentro":
            return self.crossentropy(prediction, labels)

    # src: fundamentals of machine learning lecture slides.
    def crossentropy(self, predictions, labels):
        return -1.0 * np.sum([labels*np.log(p) if p != 0 else labels*np.log(0.0000000000001) for p in predictions])
