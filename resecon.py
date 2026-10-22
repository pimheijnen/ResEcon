"""Solving a standard resource economics problem with natural growth
function h(.), utility function u(.) and discount factor delta.

The Bellman equation is

    V(x_t) = max_{0 <= x_{t+1} <= h(x_t) + x_t}  u(h(x_t) + x_t - x_{t+1}) + delta * V(x_{t+1})

and is solved by value function iteration on a grid. Python port of resecon.m.
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt


def _elementwise(f, name, test_points):
    """Return f if it works element-wise on numpy arrays, else a vectorised wrapper.

    f passes if calling it on the array test_points gives an array of the same
    shape that matches calling it on each point separately.
    """
    test_points = np.asarray(test_points, dtype=float)
    with np.errstate(all="ignore"):
        try:
            expected = np.array([f(x) for x in test_points], dtype=float)
        except Exception as e:
            raise ValueError(f"{name} cannot be evaluated at the points {test_points}") from e
        try:
            result = np.asarray(f(test_points), dtype=float)
            ok = result.shape == test_points.shape and np.allclose(result, expected, equal_nan=True)
        except Exception:
            ok = False
    if ok:
        return f
    warnings.warn(f"{name} does not work element-wise on numpy arrays; "
                  "falling back to np.vectorize, which is slower", stacklevel=3)

    def f_scalar(x):
        # Scalar functions such as math.log raise errors where numpy returns
        # nan or -inf; return nan so the solver treats the point as infeasible.
        try:
            return f(x)
        except (ValueError, ZeroDivisionError, OverflowError):
            return np.nan

    return np.vectorize(f_scalar, otypes=[float])


class ResourceExtraction:
    """Optimal extraction of a renewable resource, solved on a grid.

    Parameters
    ----------
    h : callable
        Natural growth function, vectorised over numpy arrays.
    u : callable
        Utility of consumption, vectorised over numpy arrays.
    delta : float
        Discount factor, 0 < delta < 1.
    xmin, xmax : float
        Bounds of the grid for the resource stock.
    ngrid : int
        Number of grid points (memory use is O(ngrid**2)).
    tol : float
        Convergence tolerance on the norm of successive value functions.
    max_iter : int
        Maximum number of value function iterations.

    After calling `solve()` (done automatically when needed) the solution is
    available as the arrays `xgrid`, `V`, `xnext` and `consumption`.
    """

    def __init__(self, h, u, delta, xmin, xmax, ngrid=1000, tol=1e-6, max_iter=10_000):
        if not 0 < delta < 1:
            raise ValueError("delta must lie strictly between 0 and 1")
        self.delta = delta
        self.xmin = xmin
        self.xmax = xmax
        self.ngrid = ngrid
        self.tol = tol
        self.max_iter = max_iter

        self.xgrid = np.linspace(xmin, xmax, ngrid)
        self.h = _elementwise(h, "h", self.xgrid[[0, ngrid // 2, -1]])
        self.u = _elementwise(u, "u", [0.1, 0.5, 1.0])
        self.V = None
        self.xnext = None
        self.inext = None
        self.consumption = None
        self.iterations = None

    # ------------------------------------------------------------------ solver

    @property
    def solved(self):
        return self.V is not None

    def _ensure_solved(self):
        if not self.solved:
            self.solve()

    def _utility_matrix(self):
        """U[i, j] = u(h(x_i) + x_i - x_j), or -inf if consumption is negative."""
        available = self.h(self.xgrid) + self.xgrid
        c = available[:, None] - self.xgrid[None, :]
        U = np.full(c.shape, -np.inf)
        feasible = c >= 0
        with np.errstate(divide="ignore", invalid="ignore"):
            U[feasible] = self.u(c[feasible])
        U[np.isnan(U)] = -np.inf
        return U

    def solve(self):
        """Run value function iteration. Returns self, so calls can be chained."""
        U = self._utility_matrix()
        V = U.max(axis=1) / (1 - self.delta)
        if np.any(np.isinf(V)):
            raise RuntimeError("no feasible actions")

        for it in range(1, self.max_iter + 1):
            W = V
            V = (U + self.delta * W[None, :]).max(axis=1)
            if np.linalg.norm(W - V) < self.tol:
                break
        else:
            raise RuntimeError(f"value function did not converge in {self.max_iter} iterations")

        idx = np.argmax(U + self.delta * V[None, :], axis=1)
        self.V = V
        self.inext = idx
        self.xnext = self.xgrid[idx]
        self.consumption = self.h(self.xgrid) + self.xgrid - self.xnext
        self.iterations = it
        return self

    # --------------------------------------------------------------- analysis

    def policy(self, x):
        """Optimal next-period stock at stock level(s) x (linear interpolation)."""
        self._ensure_solved()
        return np.interp(x, self.xgrid, self.xnext)

    def consumption_at(self, x):
        """Optimal consumption at stock level(s) x (linear interpolation)."""
        self._ensure_solved()
        return np.interp(x, self.xgrid, self.consumption)

    def value(self, x):
        """Value function at stock level(s) x (linear interpolation)."""
        self._ensure_solved()
        return np.interp(x, self.xgrid, self.V)

    def simulate(self, x0, T):
        """Simulate the optimal path from initial stock x0.

        Returns (path, cons_path): the resource stock in periods 1..T and the
        consumption in periods 1..T-1.
        """
        path = np.zeros(T)
        cons_path = np.zeros(T - 1)
        path[0] = x0
        for t in range(1, T):
            path[t] = self.policy(path[t - 1])
            cons_path[t - 1] = self.consumption_at(path[t - 1])
        return path, cons_path

    def _sign_changes(self, before, after):
        """Grid indices (left, right) of consecutive points with nonzero net
        growth where its sign changes from `before` to `after` (+1 or -1).
        Points with zero net growth in between are skipped.

        The direction of net growth is read from the grid indices: the stock
        grows if inext[i] > i, declines if inext[i] < i, and stays put if
        inext[i] == i. This is an exact integer comparison.
        """
        self._ensure_solved()
        s = np.sign(self.inext - np.arange(self.ngrid))
        nz = np.flatnonzero(s)
        left, right = nz[:-1], nz[1:]
        change = (s[left] == before) & (s[right] == after)
        return left[change], right[change]

    def jump_points(self):
        """Stock levels where the policy function jumps.

        Between two stable steady states there is a point where the direction
        of net growth h(x) - c(x) changes from negative to positive: below it
        the stock declines, above it the stock grows. The policy function jumps
        there. Returns an array of these points (empty if there are none).
        """
        left, right = self._sign_changes(-1, +1)
        return (self.xgrid[left] + self.xgrid[right]) / 2

    def steady_states(self):
        """Stable steady states: stock levels where net growth changes from
        positive to negative. Returns an array (accurate to about one grid
        step). The trivial steady state at x = 0 is not included.
        """
        left, right = self._sign_changes(+1, -1)
        return (self.xgrid[left] + self.xgrid[right]) / 2

    # --------------------------------------------------------------- plotting

    def plot_path(self, x0, T, fig=None):
        """Plot the simulated stock (top) and consumption (bottom) over time.

        x0 is a single starting value or a sequence of them. A single path is
        drawn in black; several paths get one colour each.
        Dotted horizontal lines mark the stock and consumption levels of each
        steady state (see `steady_states`).
        """
        starts = np.atleast_1d(x0)
        paths = [self.simulate(x, T) for x in starts]
        colors = ["k"] if len(starts) == 1 else [f"C{i % 10}" for i in range(len(starts))]
        xss = self.steady_states()
        css = self.h(xss)  # at a steady state consumption equals natural growth
        if fig is None:
            fig = plt.figure()
        fig.clf()
        ax1, ax2 = fig.subplots(2, 1)

        ax1.set_xlim(1, T)
        ax1.set_ylim(0, self.xmax)
        ax1.set_ylabel("Resource level", fontsize=18)
        for x in xss:
            ax1.plot([1, T], [x, x], "k:")

        ax2.set_xlim(1, T)
        ax2.set_ylim(0, max(cons_path.max() for _, cons_path in paths) * 1.2)
        ax2.set_ylabel("Consumption", fontsize=18)
        ax2.set_xlabel("Time ($t$)", fontsize=16)
        for c in css:
            ax2.plot([1, T], [c, c], "k:")

        for (path, cons_path), color in zip(paths, colors):
            ax1.scatter(np.arange(1, T + 1), path, color=color)
            ax2.scatter(np.arange(1, T), cons_path, color=color)

        fig.tight_layout()
        return fig

    def plot_policy(self, ax=None):
        """Plot optimal consumption against natural growth h(x).

        The consumption curve is not connected across the jump points (see
        `jump_points`). Each steady state (see `steady_states`) is marked with
        a dot, where consumption equals natural growth.
        """
        self._ensure_solved()
        if ax is None:
            ax = plt.figure().gca()
        x, c = self.xgrid, self.consumption

        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(0, c.max() * 1.2)
        ax.set_title("Consumption vs. natural growth", fontsize=18)
        ax.set_xlabel("Level of resource", fontsize=16)
        ax.plot(x, self.h(x), "k", linewidth=2)

        _, jumps = self._sign_changes(-1, +1)
        for xs, cs in zip(np.split(x, jumps), np.split(c, jumps)):
            ax.plot(xs, cs, "k", linewidth=2)

        for xss in self.steady_states():
            css = self.h(xss)  # at a steady state consumption equals natural growth
            ax.scatter([xss], [css], s=100, color="k", zorder=3)
        return ax
