import numpy as np
import matplotlib.pyplot as plt


def k_means(K, dataset, e):
    idx = np.random.choice(len(dataset), K, False)
    mi = dataset[idx, :]
    norms = [[np.linalg.norm(vector - cluster_mid) for cluster_mid in mi] for vector in dataset]
    indicies = [np.argmin(norm) for norm in norms]
    err = quant_error(dataset,indicies, mi)
    while True:
        prev_err = err
        sums = list()
        for cluster_mid in mi:
            cluster_mid = np.zeros(len(dataset[0]))
            sums.append(0)
        i = 0
        for index in indicies:
            mi[index] += dataset[i]
            sums[index] += 1
            i += 1
        for i in range(len(sums)):
            mi[i] /= sums[i]
        norms = [[np.linalg.norm(vector - cluster_mid) for cluster_mid in mi] for vector in dataset]
        indicies = [np.argmin(norm) for norm in norms]
        err = quant_error(dataset, indicies, mi)
        if abs((prev_err - err)/err) < e:
            break
    return indicies, mi

def quant_error(dataset,indicies, mi):
    err = 0
    for i in range(len(dataset)):
        err += pow(np.linalg.norm(dataset[i] - mi[indicies[i]]), 2)
    return err

def fit(dataset, K, e=0.1):
    return k_means(dataset, K, e)

def visualize(dataset, indicies, mi):
    fig, ax = plt.subplots(1, 1)
    plt.axis('equal')
    ax.scatter(x=dataset[:, 0], y=dataset[:, 1], c=indicies,s=20, cmap='viridis')
    ax.scatter(x=mi[:, 0], y=mi[:, 1], c='black', s=200, alpha=0.5)
    plt.show()



