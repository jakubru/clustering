import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import k_means



def gaussian_mixture_model(dataset, mi, indicies, K):
    means_k = mi
    c_k = []
    sigma_k = []
    clusters = list(zip(dataset, indicies))
    for i in range(K):
        c_k.append(len(list(filter(lambda el: el == i, indicies))))
    for i in range(K):
        filtered = [pair[0] for pair in filter(lambda el: el[1] == i, list(clusters))]
        cov = 0
        for element in filtered:
            x = np.matrix(element- means_k[i])
            cov += np.dot(x.transpose(),x)
        sigma_k.append(cov/c_k[i])
    pi_k = []
    for i in range(K):
        pi_k.append(c_k[i]/len(dataset))
    while True:
        r_nk = []
        old_means = means_k
        old_sigma = sigma_k
        for element in dataset:
            r_nk.append([])
            for i in range(K):
                r_nk[len(r_nk) - 1].append(pi_k[i]*stats.multivariate_normal(means_k[i], sigma_k[i]).pdf(element))
            sum_ = sum(r_nk[len(r_nk) - 1])
            for i in range(K):
                r_nk[len(r_nk) - 1][i] /= sum_
        for i in range(len(r_nk)):
            r_nk[i] = np.argmax(r_nk[i])
        means_k = []
        c_k = []
        sigma_k = []
        clusters = list(zip(dataset, r_nk))
        for i in range(K):
            c_k.append(len(list(filter(lambda el: el == i, r_nk))))
        for i in range(K):
            means_k.append(sum([pair[0] for pair in filter(lambda el: el[1] == i, clusters)]) / c_k[i])
        for i in range(K):
            filtered = [pair[0] for pair in filter(lambda el: el[1] == i, list(clusters))]
            cov = 0
            for element in filtered:
                x = np.matrix(element - means_k[i])
                cov += np.dot(x.transpose(), x)
            sigma_k.append(cov / c_k[i])
        pi_k = []
        for i in range(K):
            pi_k.append(c_k[i] / len(dataset))
        if check_equal(means_k, old_means, sigma_k, old_sigma, K):
            break
    return means_k, sigma_k

def visualize(means, covariances):
    x = np.linspace(-20, 20, 100)
    y = np.linspace(-20, 20, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    Z = np.array([np.zeros(100) for _ in range (100)])
    for i in range(len(means)):
        rv = stats.multivariate_normal(means[i], covariances[i])
        Z_tmp = rv.pdf(pos)
        for j in range(len(Z)):
            Z[j] = np.maximum(Z[j], Z_tmp[j])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(X, Y, Z)
    plt.show()





def fit(dataset, K, mi, indicies):
    return gaussian_mixture_model(dataset, mi, indicies, K)



def check_equal(means_1, means_2, sigma_1, sigma_2, K):
    for i in range(K):
        if not np.array_equal(means_1[i],  means_2[i]):
            return False
        if not np.array_equal(sigma_1[i], sigma_2[i]):
            return False
    return True