import numpy as np
import scipy.stats as stats
import k_means
import gaussian_mixture_model
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



gauss1 = stats.multivariate_normal([12, -12], [[20, 0], [0, 20]])
gauss2 = stats.multivariate_normal([6, 6], [[10, 0], [0, 10]])
gauss3 = stats.multivariate_normal([-10, 10], [[5, 0], [0, 5]])
gauss4 = stats.multivariate_normal([-12, -12], [[12, 0], [0, 12]])


dataset = []
for _ in range(400):
    dataset.append(gauss1.rvs())
for _ in range(400):
    dataset.append(gauss2.rvs())
for _ in range(400):
    dataset.append(gauss3.rvs())
for _ in range(400):
    dataset.append(gauss4.rvs())
dataset = np.array(dataset)


indicies, mi = k_means.fit(4,dataset)
#k_means.visualize(dataset, indicies, mi)

#gaussian_mixture_model.gaussian_mixture_model(dataset, indicies, 4)




gaussian_mixture_model.visualize([[12, -12], [-10, 10]], [[[20, 0], [0, 20]], [[5, 0], [0, 5]]])