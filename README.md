# `ResourceExtraction` — optimal resource extraction in Python

Python port of `resecon.m`. Solves

$$V(x_t) = \max_{0 ≤ x_{t+1} ≤ h(x_t) + x_t}  u(h(x_t) + x_t − x_{t+1}) + δ V(x_{t+1})$$

by value function iteration on a grid. Requires `numpy` and `matplotlib`.

## Quick start

```python
import numpy as np
from resecon import ResourceExtraction

model = ResourceExtraction(h=lambda x: x * (1 - x), u=np.sqrt, delta=0.8,
                           xmin=0, xmax=1, ngrid=1000).solve()

path, cons = model.simulate(x0=0.9, T=10)
model.plot_path(x0=0.9, T=10)
model.plot_policy()
```

See `logistic_growth.py` and `shallowlake.py` for the two original examples.

## Constructor

`ResourceExtraction(h, u, delta, xmin, xmax, ngrid=1000, tol=1e-6, max_iter=10000)`

| Argument | Meaning |
|---|---|
| `h` | Natural growth function. Works fastest element-wise on numpy arrays, but scalar functions are accepted too (see below). |
| `u` | Utility of consumption. Same as `h`. |
| `delta` | Discount factor, strictly between 0 and 1. |
| `xmin`, `xmax` | Bounds of the stock grid. |
| `ngrid` | Number of grid points. Memory use grows with `ngrid**2`. |
| `tol` | Stopping tolerance for the norm of successive value functions. |
| `max_iter` | Maximum number of iterations. Raises `RuntimeError` if this is reached. |

### Scalar-only functions

The constructor checks whether `h` and `u` work element-wise. It calls each function on a small test array and compares the result with calling it point by point. `h` is tested at the two ends and the middle of the grid, and `u` at 0.1, 0.5 and 1. A function fails the check if it raises an error on an array, returns the wrong shape, or gives different values. Examples are `math.sqrt` or a function with an `if x < ...` statement. When that happens, the class issues a warning and wraps the function in `np.vectorize`. The results are the same but slower: about twice as slow for `ngrid=1000`.

Inside the wrapper, a point where the function raises `ValueError`, `ZeroDivisionError` or `OverflowError` (e.g. `math.log(0)`) counts as infeasible. MATLAB treated complex values the same way. If the function fails even at the individual test points, the constructor raises a `ValueError`.

## Methods

| Method | Returns |
|---|---|
| `solve()` | Runs value function iteration and returns `self`, so you can write `ResourceExtraction(...).solve()`. The other methods call it automatically if needed. |
| `policy(x)` | Optimal next-period stock at `x` (interpolated; scalar or array). |
| `consumption_at(x)` | Optimal consumption at `x`. |
| `value(x)` | Value function at `x`. |
| `simulate(x0, T)` | `(path, cons_path)`: the stock in periods 1..T and consumption in periods 1..T−1. |
| `plot_path(x0, T, fig=None)` | Two panels: stock and consumption over time. `x0` is one starting value or a list of them, e.g. `x0=[0.35, 0.6]`. A single path is drawn in black. Several paths each get their own colour. Dotted lines mark the stock and consumption levels of each steady state. |
| `jump_points()` | Stock levels where the policy function jumps (see below). Returns an empty array if there are none. |
| `steady_states()` | Stable steady states, where net growth changes from + to −, accurate to about one grid step. The trivial steady state at x = 0 is not included. |
| `plot_policy(ax=None)` | Plots consumption against natural growth h(x). The consumption curve is always left unconnected at the jump points. Each steady state is marked with a dot on h(x). |

The plot methods return the matplotlib figure or axes so you can customise them further.

## Steady states and jumps in the policy function

With multiple steady states, the policy function can jump. Under the optimal policy, net growth is xnext − x = h(x) − c(x). At a stable steady state it changes sign from + to −. At a jump it changes from − to +: below that point the stock declines, and above it the stock grows. `steady_states()` and `jump_points()` find these two kinds of sign changes on the grid. Each returns the midpoint between the last grid point before the change and the first one after it. The optimal next stock is always a grid point, `xgrid[inext[i]]`, so the direction of net growth is simply the sign of `inext[i] − i`. This is an exact integer comparison, and no tolerance is needed. Grid points with `inext[i] == i` are skipped. In the shallow-lake example this gives steady states at x ≈ 0.099 and x ≈ 0.707 and a jump at x ≈ 0.443.

## Attributes after solving

`xgrid`, `V`, `xnext`, `consumption`: the same four outputs as `resecon.m`, as 1-D numpy arrays. `inext` holds the grid index of the optimal next stock, so `xnext == xgrid[inext]`. `iterations` holds the number of iterations used.

## Differences from the MATLAB version

- A choice is infeasible when consumption would be negative, and its utility is set to −∞. MATLAB instead detected infeasible choices from complex values of `sqrt`. The result is the same for `sqrt`, and this rule also works for other utility functions such as `log`.
- Interpolation uses `np.interp`, which holds the end values outside the grid instead of returning `NaN` as `interp1` does.
- The solver vectorises the utility matrix instead of using `arrayfun`. It runs in under a second for `ngrid=1000`.
