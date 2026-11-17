import numpy as np
import matplotlib.pyplot as plt

from resecon import ResourceExtraction

pn = 25 * np.polyint(np.polymul(np.polymul([1, -1 / 5], [-1, 1 / 4]), [1, -3 / 4]))

temp = np.roots(pn)
xmax = temp[np.abs(temp.imag) < 1e-12].real.max()

delta = 0.8
h = lambda x: np.polyval(pn, x)
u = np.sqrt

model = ResourceExtraction(h, u, delta, xmin=0, xmax=xmax, ngrid=1000).solve()

model.plot_path(x0=[0.35,0.5], T=15)
model.plot_policy()
plt.show()
