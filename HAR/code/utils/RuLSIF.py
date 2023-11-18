from scipy.stats import norm
from densratio.core import densratio
from matplotlib import pyplot as plt

x = norm.rvs(size = 200, loc = 1, scale = 1./8)
y = norm.rvs(size = 200, loc = 1, scale = 1./2) # loc is the average, scale is deivation
result = densratio(x, y)


density_ratio = result.compute_density_ratio(y)

plt.plot(y, density_ratio, "o")
plt.xlabel("x")
plt.ylabel("Density Ratio")
plt.show()