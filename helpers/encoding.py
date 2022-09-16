import numpy as np

def one_hot_encoding(labels):
    unique_labels = np.unique(labels)
    newlabels = np.zeros([len(labels), len(unique_labels)])
    for index in range(0, len(newlabels)):
        newlabels[index, np.argwhere(unique_labels==labels[index])[0]] = 1
    return unique_labels, newlabels
