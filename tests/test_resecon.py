import math

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from resecon import ResourceExtraction


def logistic(x):
    return x * (1 - x)


@pytest.mark.parametrize("delta", [0.8, 0.95])
def test_logistic_steady_state_matches_golden_rule(delta):
    # With logistic growth the interior steady state solves h'(x) = 1/delta - 1.
    model = ResourceExtraction(logistic, np.sqrt, delta, xmin=0, xmax=1).solve()
    expected = (1 - (1 / delta - 1)) / 2
    (xss,) = model.steady_states()
    assert xss == pytest.approx(expected, abs=2 / model.ngrid)
    assert model.jump_points().size == 0


def test_solution_arrays_are_consistent():
    model = ResourceExtraction(logistic, np.sqrt, 0.8, xmin=0, xmax=1, ngrid=200).solve()
    assert np.array_equal(model.xnext, model.xgrid[model.inext])
    assert np.allclose(model.consumption, logistic(model.xgrid) + model.xgrid - model.xnext)
    assert np.all(model.consumption >= 0)


def test_shallow_lake_has_two_steady_states_and_a_jump():
    pn = 25 * np.polyint(np.polymul(np.polymul([1, -1 / 5], [-1, 1 / 4]), [1, -3 / 4]))
    roots = np.roots(pn)
    xmax = roots[np.abs(roots.imag) < 1e-12].real.max()
    model = ResourceExtraction(lambda x: np.polyval(pn, x), np.sqrt, 0.8,
                               xmin=0, xmax=xmax).solve()
    assert model.steady_states() == pytest.approx([0.099, 0.707], abs=0.005)
    assert model.jump_points() == pytest.approx([0.443], abs=0.005)


def test_simulate_converges_to_steady_state():
    model = ResourceExtraction(logistic, np.sqrt, 0.8, xmin=0, xmax=1)
    path, cons = model.simulate(x0=0.9, T=40)
    assert path.shape == (40,) and cons.shape == (39,)
    assert path[-1] == pytest.approx(model.steady_states()[0], abs=0.01)


def test_scalar_functions_are_wrapped_with_warning():
    def h(x):
        return x * (1 - x) if x < 1 else 0.0

    with pytest.warns(UserWarning):
        model = ResourceExtraction(h, math.sqrt, 0.8, xmin=0, xmax=1, ngrid=200)
    reference = ResourceExtraction(logistic, np.sqrt, 0.8, xmin=0, xmax=1, ngrid=200)
    assert np.allclose(model.solve().V, reference.solve().V)


def test_invalid_delta():
    with pytest.raises(ValueError):
        ResourceExtraction(logistic, np.sqrt, 1.0, xmin=0, xmax=1)


def test_log_utility_needs_positive_xmin():
    with pytest.raises(RuntimeError, match="xmin > 0"):
        ResourceExtraction(logistic, np.log, 0.8, xmin=0, xmax=1).solve()
    model = ResourceExtraction(logistic, np.log, 0.8, xmin=1e-3, xmax=1).solve()
    assert model.steady_states() == pytest.approx([0.375], abs=0.002)


def test_plots_run():
    model = ResourceExtraction(logistic, np.sqrt, 0.8, xmin=0, xmax=1, ngrid=200)
    model.plot_path(x0=[0.2, 0.9], T=10)
    model.plot_policy()
