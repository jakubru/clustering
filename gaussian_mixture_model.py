import numpy as np
import matplotlib.pyplot as plt


def gaussian_mixture_model(dataset, indicies, mi, K):
    means_k = []
    c_k = []
    sigma_k = []
    clusters = zip(dataset, indicies)
    for i in range(K):
        c_k.append(len(list(filter(lambda el: el == i, indicies))))
    for i in range (K):
        means_k.append(sum([pair[0] for pair in filter(lambda el: el[1] == i, clusters)])/c_k[i])
    for i in range(K):
        filtered = [pair[0] for pair in filter(lambda el: el[1] == i, clusters)]
        cov = 0
        print(filtered)
        for element in filtered:
            cov += (element - means_k[i])*(element - means_k[i]).tanspose()
            print(cov)
        cov = cov/c_k[i]
        sigma_k.append(cov)
    print(sigma_k)


