import numpy as np
import random
import matplotlib.pyplot as plt


def k_means(K, dataset, e=0.001):
    idx = np.random.choice(len(dataset), K, False)
    mi = dataset[idx, :]
    norms = [[np.linalg.norm(vector - cluster_mid) for cluster_mid in mi] for vector in dataset]
    indicies = [np.argmin(norm) for norm in norms]
    clustering = [(norms[i][indicies[i]]) for i in range(len(indicies))]
    err = quant_error(clustering)
    while True:
        prev_err = err
        sums = list()
        for cluster_mid in mi:
            cluster_mid = np.zeros(len(dataset[0]))
            sums.append(0)
        i = 0
        for index in indicies:
            mi[index] += norms[i][index]
            sums[index] += 1
            i += 1
        for i in range(len(sums)):
            mi[i] /= sums[i]
        norms = [[np.linalg.norm(vector - cluster_mid) for cluster_mid in mi] for vector in dataset]
        indicies = [np.argmin(norm) for norm in norms]
        clustering = [(norms[i][indicies[i]]) for i in range(len(indicies))]
        err = quant_error(clustering)
        print(mi)
        if abs((prev_err - err)/err) < e:
            break
    return mi

def quant_error(clustering):
    return sum(clustering) / len(clustering)

def fit():
    pass

def visualize():
    pass



