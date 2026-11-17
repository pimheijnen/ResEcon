# resecon: optimal extraction of a renewable resource

`resecon` solves the textbook problem of renewable resource economics. A
planner owns a stock of a renewable resource (fish, forest, groundwater,
a clean lake) that grows naturally. In each period the planner decides how much
to harvest and consume, and how much to leave to grow. The package computes the
optimal harvesting policy, the value function and the steady states for any
growth function and utility function you supply. It also simulates and plots
the optimal paths.

The solver uses value function iteration on a grid. It is slower than methods
that solve first-order conditions, but it needs no concavity or smoothness. This
makes it suitable for non-convex problems: growth with an Allee effect or a
tipping point, multiple steady states, and policies that jump.

## Installation

`resecon` needs Python 3.9 or newer, with `numpy` and `matplotlib`. They are
installed automatically. Install the package straight from GitHub with

```bash
pip install git+https://github.com/pimheijnen/ResEcon.git
```

To upgrade to the latest version later, add `--upgrade` to the same command.

## Quick start

```python
import numpy as np
from resecon import ResourceExtraction

model = ResourceExtraction(h=lambda x: x * (1 - x),   # logistic growth
                           u=np.sqrt,                 # utility of consumption
                           delta=0.8,                 # discount factor
                           xmin=0, xmax=1, ngrid=1000).solve()

model.steady_states()            # array([0.37487487])
path, cons = model.simulate(x0=0.9, T=10)
model.plot_path(x0=0.9, T=10)
model.plot_policy()
```

## The optimisation problem

### The model

Let $x_t \ge 0$ be the resource stock at the start of period $t$. Left alone,
the stock grows by $h(x_t)$ in that period. The function $h$ is called the
natural growth function, and it may be negative. The planner then harvests and
consumes $c_t$, and what remains is next period's stock:

```math
x_{t+1} = x_t + h(x_t) - c_t .
```

Consumption gives utility $u(c_t)$, and future utility is discounted by the
factor $\delta \in (0, 1)$. Starting from a given stock $x_0$, the planner solves

```math
\max_{\{c_t\}_{t \ge 0}} \; \sum_{t=0}^{\infty} \delta^t \, u(c_t)
\quad \text{subject to} \quad
x_{t+1} = x_t + h(x_t) - c_t, \qquad c_t \ge 0, \qquad x_{t+1} \ge 0 .
```

The package works with the next-period stock $x_{t+1}$ as the choice
variable, rather than with $c_t$. Consumption is then whatever is left over,
$c_t = x_t + h(x_t) - x_{t+1}$. The value function $V$ satisfies the Bellman
equation

```math
V(x) = \max_{0 \,\le\, x' \,\le\, x + h(x)} \; u\big(x + h(x) - x'\big) + \delta \, V(x') .
```

You supply $h$, $u$ and $\delta$. The package returns $V$, the policy function
$x'(x)$ and the consumption function $c(x) = x + h(x) - x'(x)$.

### Assumptions built into this formulation

These assumptions come from the model itself, not from the numerical method.

- **One stock, discrete time, infinite horizon.** There is a single resource
  and time runs forever in whole periods.
- **Deterministic.** Growth is known exactly, and there are no shocks.
- **Stationary.** $h$, $u$ and $\delta$ do not change over time.
- **Growth comes before harvest.** In each period the stock first grows by
  $h(x_t)$. The harvest is then taken from $x_t + h(x_t)$. So the harvest in
  period $t$ can be at most $x_t + h(x_t)$.
- **Utility depends on the harvest only.** The stock itself gives no utility
  (no amenity or existence value), and harvesting costs do not depend on the
  stock. Stock-dependent harvesting costs, as in the Gordon–Schaefer model,
  cannot be represented.
- **Harvest is irreversible.** $c_t \ge 0$, so the stock cannot be restocked
  above what natural growth provides.
- **A single decision maker.** The solution is the sole-owner or social-planner
  optimum. Open access and strategic interaction between several harvesters are
  not covered.

$u$ does not have to be concave, and $h$ does not have to be concave or smooth.
The function $h$ only has to be defined on the grid.

### How it is solved

1. The stock is restricted to a grid of `ngrid` evenly spaced points
   between `xmin` and `xmax`. Both the current stock $x$ and the choice $x'$
   lie on this grid.
2. For each pair of grid points $(x_i, x_j)$ the package computes the utility
   $u(x_i + h(x_i) - x_j)$. A pair is infeasible if consumption would be
   negative or if $u$ returns `nan`. It then gets utility $-\infty$.
3. Starting from $V_0(x_i) = \max_j u(\cdot)/(1 - \delta)$, the Bellman
   operator is applied until $\lVert V_{k} - V_{k-1} \rVert_2 <$ `tol`. The
   norm is the Euclidean norm over all grid points.
4. The optimal choice at each grid point is the maximising grid point $x_j$.
   Values between grid points are found by linear interpolation.

## Limitations

The results are those of a *discretised* problem. Most limitations follow from
that, and it is worth knowing what they are before you interpret the output.

**The stock is confined to `[xmin, xmax]`.** The next-period stock must be a
grid point, so the grid bounds act as constraints of the problem.

- If `xmax` is too small, the planner cannot let the stock grow beyond it.
  Anything above `xmax` must be consumed. The true steady state is then cut
  off. The stock settles at `xmax` and `steady_states()` returns nothing. For
  example, with logistic growth and `delta=0.95` the steady state is about
  0.474. With `xmax=0.4`, the model silently keeps the stock at 0.4. Choose
  `xmax` at or above the carrying capacity, and check that `xnext` does not
  press against `xmax`.
- If `xmin > 0`, the stock can never fall below `xmin`. So extinction or full
  depletion is ruled out by construction. Use `xmin=0` if depletion is a
  relevant outcome.

**Utility that is unbounded at zero needs `xmin > 0`.** With `u = np.log` (or
CRRA utility with curvature of at least 1) and `xmin=0`, the only feasible
choice at $x = 0$ is zero consumption, so its utility is $-\infty$.
`solve()` then raises a `RuntimeError`. Set `xmin` to a small positive number
instead, e.g. `xmin=1e-3`.

**Accuracy is limited by the grid.** The optimal next stock is always a grid
point, so the policy, the steady states and the jump points are accurate to
about one grid step, `(xmax - xmin) / (ngrid - 1)`. Consumption is computed
exactly given that choice. At grid points the computed value is, up to the
stopping tolerance, no higher than the true value, because only a subset of
choices is available. Refine the grid if you need more precision.

**Between grid points everything is interpolated.** `policy`, `consumption_at`
and `value` interpolate linearly between grid points. Near a jump point the
interpolated policy mixes the two branches on either side of the jump, and it is
not optimal there. `simulate` uses these interpolated functions, so a path that
starts within a grid step of a jump point is unreliable. Outside `[xmin, xmax]`
the interpolation holds the end values.

**Memory and time grow with `ngrid**2`.** The solver stores several
`ngrid × ngrid` arrays: about 8 MB each for `ngrid=1000`, and 800 MB each for
`ngrid=10000`. The number of iterations grows as `delta` approaches 1. Logistic
growth with `ngrid=1000` takes 74 iterations (under 0.1 s) for `delta=0.8`,
323 iterations (0.3 s) for `delta=0.95` and 1649 iterations (about 2 s) for
`delta=0.99`. `ngrid=3000` with `delta=0.8` takes about 1.5 s. If the limit
`max_iter` is reached, `solve()` raises a `RuntimeError`.

**The stopping rule uses the Euclidean norm.** The tolerance applies to the
norm of the change over all grid points. So the same `tol` is stricter on a
finer grid. It does not bound the error relative to the true value function.

**Steady state detection is limited.** `steady_states()` reports only stable
interior steady states. The trivial steady state at $x = 0$ is not included,
and neither is a steady state at the edge of the grid. Unstable steady states
are not reported as such. Instead, `jump_points()` reports where the direction
of the optimal path changes from decline to growth. This is the threshold
between basins of attraction (a Skiba point), not necessarily an unstable
steady state of the natural dynamics.

**No checks on the economics.** The package does not check that $u$ is
increasing, or that $h$ describes a sensible growth process. It solves whatever
problem it is given.

## Examples

The `examples/` folder contains three scripts.

| Script | Model |
|---|---|
| `logistic_growth.py` | Logistic growth $h(x) = x(1-x)$ and square-root utility. One steady state that every path converges to. |
| `shallowlake.py` | A shallow lake with a cubic growth function. Two stable steady states and a jump in the policy function between them. |
| `allee_fishery.py` | A fishery with an Allee effect. Below a tipping point it is optimal to fish the stock out, even where it would recover on its own. |

Download the folder from GitHub and run, for example,
`python examples/shallowlake.py`.

## Reference

### Constructor

`ResourceExtraction(h, u, delta, xmin, xmax, ngrid=1000, tol=1e-6, max_iter=10000)`

| Argument | Meaning |
|---|---|
| `h` | Natural growth function. Works fastest element-wise on numpy arrays, but scalar functions are accepted too (see below). |
| `u` | Utility of consumption. Same as `h`. |
| `delta` | Discount factor, strictly between 0 and 1. |
| `xmin`, `xmax` | Bounds of the stock grid. These are also bounds on the stock in the problem (see [Limitations](#limitations)). |
| `ngrid` | Number of grid points. Memory use grows with `ngrid**2`. |
| `tol` | Stopping tolerance for the norm of successive value functions. |
| `max_iter` | Maximum number of iterations. Raises `RuntimeError` if this is reached. |

#### Scalar-only functions

The constructor checks whether `h` and `u` work element-wise. It calls each
function on a small test array and compares the result with calling it point by
point. `h` is tested at the two ends and the middle of the grid, and `u` at 0.1,
0.5 and 1. A function fails the check if it raises an error on an array, returns
the wrong shape, or gives different values. Examples are `math.sqrt` or a
function with an `if x < ...` statement. When that happens, the class issues a
warning and wraps the function in `np.vectorize`. The results are the same but
slower: about twice as slow for `ngrid=1000`.

Inside the wrapper, a point where the function raises `ValueError`,
`ZeroDivisionError` or `OverflowError` (e.g. `math.log(0)`) counts as
infeasible. If the function fails even at the individual test points, the
constructor raises a `ValueError`.

### Methods

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
| `plot_policy(ax=None)` | Plots consumption against natural growth h(x). The consumption curve is left unconnected at the jump points. Each steady state is marked with a dot on h(x). |

The plot methods return the matplotlib figure or axes so you can customise them
further.

### Attributes after solving

| Attribute | Contents |
|---|---|
| `xgrid` | The grid of stock levels. |
| `V` | Value function on the grid. |
| `xnext` | Optimal next-period stock at each grid point. |
| `consumption` | Optimal consumption at each grid point. |
| `inext` | Grid index of the optimal next stock, so `xnext == xgrid[inext]`. |
| `iterations` | Number of iterations used. |

### Steady states and jumps in the policy function

With multiple steady states, the policy function can jump. Under the optimal
policy, net growth is $x' - x = h(x) - c(x)$. At a stable steady state it
changes sign from + to −. At a jump it changes from − to +: below that point
the stock declines, and above it the stock grows. `steady_states()` and
`jump_points()` find these two kinds of sign changes on the grid. Each returns
the midpoint between the last grid point before the change and the first one
after it. The optimal next stock is always a grid point, `xgrid[inext[i]]`, so
the direction of net growth is simply the sign of `inext[i] − i`. This is an
exact integer comparison, and no tolerance is needed. Grid points with
`inext[i] == i` are skipped. In the shallow-lake example this gives steady
states at x ≈ 0.099 and x ≈ 0.707 and a jump at x ≈ 0.443.

## Coming from the MATLAB version

`resecon` started as a port of the MATLAB script `resecon.m`. The attributes
`xgrid`, `V`, `xnext` and `consumption` are its four outputs. The differences:

- A choice is infeasible when consumption would be negative, and its utility is
  set to −∞. MATLAB instead detected infeasible choices from complex values of
  `sqrt`. The result is the same for `sqrt`, and this rule also works for other
  utility functions such as `log`.
- Interpolation uses `np.interp`, which holds the end values outside the grid
  instead of returning `NaN` as `interp1` does.
- The solver vectorises the utility matrix instead of using `arrayfun`. It solves
  `ngrid=1000` in well under a second unless `delta` is close to 1.

## Development

```bash
git clone https://github.com/pimheijnen/ResEcon.git
cd ResEcon
pip install -e ".[test]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
