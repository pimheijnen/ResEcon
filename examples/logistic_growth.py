import numpy as np
import matplotlib.pyplot as plt

from resecon import ResourceExtraction

delta = 0.8
h = lambda x: x * (1 - x)
u = np.sqrt

model = ResourceExtraction(h, u, delta, xmin=0, xmax=1, ngrid=1000).solve()

model.plot_path(x0=0.9, T=10)
model.plot_policy()
plt.show()
