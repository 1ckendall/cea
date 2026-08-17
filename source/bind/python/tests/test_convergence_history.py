import numpy as np
import pytest

import cea


def _case(history=False):
    reactants = cea.Mixture(["H2", "O2"])
    products = cea.Mixture(["H", "H2", "H2O", "O", "O2", "OH"])
    solver = cea.EqSolver(products, reactants=reactants)
    solution = cea.EqSolution(solver, history=history)
    solver.solve(solution, cea.TP, 3000.0, 1.01325, np.array([2.0 / 3.0, 1.0 / 3.0]))
    return solution


def test_history_is_opt_in():
    assert _case(history=False).convergence_history == []


def test_history_exposes_newton_state_and_species():
    solution = _case(history=True)
    history = solution.convergence_history
    assert solution.converged
    assert len(history) >= 1
    assert history[-1]["temperature"] == 3000.0
    assert history[-1]["flags"]["converged"]
    assert set(history[-1]["species_moles"]) == {"H", "H2", "H2O", "O", "O2", "OH"}
    assert history[-1]["residual"] >= 0.0


@pytest.mark.parametrize("trace", [1.0e-6, 1.0e-10, 1.0e-14])
def test_history_trace_sweep_is_finite_and_normalized(trace):
    reactants = cea.Mixture(["CH4", "O2"])
    products = cea.Mixture(["CH4", "O2"], products_from_reactants=True)
    solver = cea.EqSolver(products, reactants=reactants, trace=trace, ions=False)
    solution = cea.EqSolution(solver, history=True)
    weights = reactants.moles_to_weights(np.array([1.0, 2.0]))

    solver.solve(solution, cea.TP, 4500.0, 0.01, weights)

    assert solution.converged
    assert np.isfinite(solution.T)
    assert np.all(np.isfinite(solution.nj))
    assert sum(solution.mole_fractions.values()) == pytest.approx(1.0, abs=1.0e-10)
    assert solution.convergence_history
