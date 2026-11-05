"""The last fish: a fishery with an Allee effect.

Many fish only breed well in a large enough school. Below the Allee
threshold A the stock shrinks even if nobody fishes:

    h(x) = r * x * (x / A - 1) * (1 - x)

Above A the stock grows towards the carrying capacity 1. The impatient
fisher (delta = 0.7) has to choose between waiting for a small stock to
recover and eating it now. The model shows a tipping point a little above A.
Below it, the fisher fishes the stock out, even when it would recover on
its own. Above it, the stock is steered to a healthy steady state.
"""

import numpy as np
import matplotlib.pyplot as plt

from resecon import ResourceExtraction

r = 0.8      # growth rate
A = 0.2      # Allee threshold
delta = 0.7  # discount factor: an impatient fisher

h = lambda x: r * x * (x / A - 1) * (1 - x)
u = np.sqrt

model = ResourceExtraction(h, u, delta, xmin=0, xmax=1, ngrid=1000).solve()

print(f"Allee threshold:  x = {A}")
print(f"Tipping point:    x = {model.jump_points()[0]:.3f}")
print(f"Steady state:     x = {model.steady_states()[0]:.3f}")

# One start below the Allee threshold, one between the threshold and the
# tipping point (it could recover, but is fished out), and two above.
model.plot_path(x0=[0.15, 0.21, 0.3, 0.9], T=12)
model.plot_policy()
plt.show()
