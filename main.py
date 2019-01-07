import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import k_means

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



print(k_means.k_means(5,dataset))


fig, ax = plt.subplots(1, 1)
plt.axis('equal')
ax.scatter(x=dataset[:,0], y=dataset[:,1])
plt.show()