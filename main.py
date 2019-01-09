import numpy as np
import scipy.stats as stats
import k_means
import gaussian_mixture_model

gauss1 = stats.multivariate_normal([0, 0], [[20, 0], [0, 20]])
gauss2 = stats.multivariate_normal([12, 12], [[3, 0], [0, 3]])
gauss3 = stats.multivariate_normal([-12, 12], [[3, 0], [0, 3]])

dataset = []
for _ in range(600):
    dataset.append(gauss1.rvs())
for _ in range(200):
    dataset.append(gauss2.rvs())
for _ in range(200):
    dataset.append(gauss3.rvs())
dataset = np.array(dataset)


indicies, mi = k_means.fit(3,dataset)
#k_means.visualize(dataset, indicies, mi)

gaussian_mixture_model.gaussian_mixture_model(dataset, indicies, mi, 3)