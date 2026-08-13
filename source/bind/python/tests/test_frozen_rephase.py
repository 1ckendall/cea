import numpy as np
import pytest


@pytest.fixture
def aluminum_oxygen_case(cea_module):
    reactants = cea_module.Mixture(["AL(cr)", "O2(L)"])
    products = cea_module.Mixture(
        ["AL(cr)", "O2(L)"], products_from_reactants=True
    )
    weights = reactants.of_ratio_to_weights(
        np.array([0.0, 1.0]), np.array([1.0, 0.0]), 0.8896
    )
    hc = (
        reactants.calc_property(
            cea_module.ENTHALPY, weights, np.array([298.15, 90.17])
        )
        / cea_module.R
    )
    return reactants, products, weights, hc


def _solve(cea_module, case, frozen_rephase):
    reactants, products, weights, hc = case
    solver = cea_module.RocketSolver(
        products, reactants=reactants, frozen_rephase=frozen_rephase
    )
    solution = cea_module.RocketSolution(solver)
    solver.solve(
        solution,
        weights,
        68.9476,
        supar=[10.0, 100.0, 1000.0],
        hc=hc,
        n_frz=1,
    )
    return solution


def test_frozen_rephase_is_opt_in(cea_module, aluminum_oxygen_case):
    with pytest.warns(RuntimeWarning, match="CEA_NOT_CONVERGED"):
        solution = _solve(cea_module, aluminum_oxygen_case, False)

    assert not solution.converged
    assert solution.num_pts == 4
    assert solution.mole_fractions["AL2O3(L)"][-1] > 0.0
    assert solution.mole_fractions["AL2O3(a)"][-1] == 0.0


def test_frozen_rephase_preserves_formula_amount(cea_module, aluminum_oxygen_case):
    solution = _solve(cea_module, aluminum_oxygen_case, True)

    assert solution.converged
    assert solution.num_pts == 5
    liquid = solution.mole_fractions["AL2O3(L)"]
    solid = solution.mole_fractions["AL2O3(a)"]
    np.testing.assert_allclose(liquid[0] + solid[0], liquid[-1] + solid[-1])
    for name, fractions in solution.mole_fractions.items():
        if name not in {"AL2O3(L)", "AL2O3(a)"}:
            assert fractions[-1] == pytest.approx(fractions[0], abs=1.0e-14)
    assert liquid[0] > 0.0
    assert liquid[-1] == 0.0
    assert solid[-1] > 0.0
    assert solution.T[-1] == pytest.approx(2293.64, abs=0.1)
    assert solution.Isp_vacuum[-1] == pytest.approx(2852.02, abs=0.1)
