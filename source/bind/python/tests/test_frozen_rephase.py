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


def _solve(cea_module, case, frozen_rephase=None):
    reactants, products, weights, hc = case
    kwargs = {"reactants": reactants}
    if frozen_rephase is not None:
        kwargs["frozen_rephase"] = frozen_rephase
    solver = cea_module.RocketSolver(products, **kwargs)
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


def test_frozen_rephase_false_matches_omitted(cea_module, aluminum_oxygen_case):
    with pytest.warns(RuntimeWarning, match="CEA_NOT_CONVERGED"):
        omitted = _solve(cea_module, aluminum_oxygen_case)
    with pytest.warns(RuntimeWarning, match="CEA_NOT_CONVERGED"):
        disabled = _solve(cea_module, aluminum_oxygen_case, False)

    assert omitted.converged == disabled.converged
    assert omitted.num_pts == disabled.num_pts
    for name in (
        "T",
        "P",
        "enthalpy",
        "entropy",
        "gamma_s",
        "Mach",
        "sonic_velocity",
        "Isp_vacuum",
        "nj",
    ):
        np.testing.assert_array_equal(getattr(omitted, name), getattr(disabled, name))
    for name in omitted.mole_fractions:
        np.testing.assert_array_equal(
            omitted.mole_fractions[name], disabled.mole_fractions[name]
        )
        np.testing.assert_array_equal(
            omitted.mass_fractions[name], disabled.mass_fractions[name]
        )


def test_frozen_rephase_preserves_formula_amount(cea_module, aluminum_oxygen_case):
    solution = _solve(cea_module, aluminum_oxygen_case, True)

    assert solution.converged
    assert solution.num_pts == 5
    names = list(solution.mole_fractions)
    liquid_idx = names.index("AL2O3(L)")
    solid_idx = names.index("AL2O3(a)")
    liquid = solution.nj[:, liquid_idx]
    solid = solution.nj[:, solid_idx]
    np.testing.assert_allclose(liquid + solid, liquid[0] + solid[0], atol=1.0e-14)
    frozen_indices = [
        i for i, name in enumerate(names) if name not in {"AL2O3(L)", "AL2O3(a)"}
    ]
    np.testing.assert_allclose(
        solution.nj[:, frozen_indices] - solution.nj[0, frozen_indices],
        0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(solution.entropy, solution.entropy[0], atol=1.0e-10)
    np.testing.assert_allclose(
        sum(solution.mole_fractions.values()), 1.0, atol=1.0e-14
    )
    np.testing.assert_allclose(
        sum(solution.mass_fractions.values()), 1.0, atol=1.0e-14
    )
    assert liquid[0] > 0.0
    assert liquid[-1] == 0.0
    assert solid[-1] > 0.0
    assert solution.T[-1] == pytest.approx(2293.64, abs=0.1)
    assert solution.Isp_vacuum[-1] == pytest.approx(2852.02, abs=0.1)


def test_frozen_rephase_without_partner_retains_partial_stop(
    cea_module, aluminum_oxygen_case
):
    reactants, products, weights, hc = aluminum_oxygen_case
    products_without_solid = cea_module.Mixture(
        [name for name in products.species_names if name != "AL2O3(a)"]
    )
    case = reactants, products_without_solid, weights, hc

    with pytest.warns(RuntimeWarning, match="CEA_NOT_CONVERGED"):
        solution = _solve(cea_module, case, True)

    assert not solution.converged
    assert solution.num_pts == 4
    assert solution.mole_fractions["AL2O3(L)"][-1] > 0.0


@pytest.fixture
def silica_case(cea_module):
    reactants = cea_module.Mixture(["N2", "SiO2(L)"])
    products = cea_module.Mixture(
        ["N2", "SiO2(a-qz)", "SiO2(b-qz)", "SiO2(b-crt)", "SiO2(L)"]
    )
    return reactants, products, np.array([0.5, 0.5])


def _silica_solve(cea_module, case, **schedule):
    reactants, products, weights = case
    solver = cea_module.RocketSolver(products, reactants=reactants, frozen_rephase=True)
    solution = cea_module.RocketSolution(solver)
    solver.solve(solution, weights, 10.0, tc=2200.0, n_frz=-1, **schedule)
    return solution


@pytest.mark.parametrize("ratio", [50.0, 6000.0])
def test_silica_against_integrated_heat_capacity(cea_module, silica_case, ratio):
    solution = _silica_solve(cea_module, silica_case, pi_p=[ratio])
    assert solution.converged
    assert solution.num_pts == 3
    temperature = solution.T[-1]
    assert 300.0 < temperature < 2200.0

    # Integrate Cp and Cp/T independently of the solver's H/S reference offsets.
    # Mixture.calc_property returns J/(kg K); each component has mass fraction 0.5.
    nodes, quadrature_weights = np.polynomial.legendre.leggauss(48)
    phases = [
        ("N2", 300.0, 1000.0),
        ("N2", 1000.0, 2200.0),
        ("SiO2(a-qz)", 300.0, 848.0),
        ("SiO2(b-qz)", 848.0, 1200.0),
        ("SiO2(b-crt)", 1200.0, 1996.0),
        ("SiO2(L)", 1996.0, 2200.0),
    ]
    delta_h = delta_s = 0.0
    for name, lower, upper in phases:
        lower = max(lower, temperature)
        if lower >= upper:
            continue
        species = cea_module.Mixture([name])
        midpoint = (lower + upper) / 2
        half_width = (upper - lower) / 2
        temperatures = midpoint + half_width * nodes
        cp = np.array([
            species.calc_property(cea_module.FROZEN_CP, np.array([1.0]), t)
            for t in temperatures
        ])
        delta_h -= 0.5 * half_width * np.dot(quadrature_weights, cp)
        delta_s -= 0.5 * half_width * np.dot(quadrature_weights, cp / temperatures)

    assert solution.enthalpy[-1] == pytest.approx(
        solution.enthalpy[0] + delta_h / 1000, abs=2.0e-5
    )
    gas_amount = solution.nj[0, list(solution.mole_fractions).index("N2")]
    assert delta_s == pytest.approx(-gas_amount * cea_module.R * np.log(ratio), abs=1.0e-5)
    np.testing.assert_array_equal(solution.nj[:, 0], solution.nj[0, 0])
    np.testing.assert_allclose(solution.nj[:, 1:].sum(axis=1), solution.nj[0, 1:].sum(), atol=1e-15)


@pytest.mark.parametrize("area", [10.0, 100.0, 1000.0])
def test_silica_area_and_pressure_schedules_agree(cea_module, silica_case, area):
    by_area = _silica_solve(cea_module, silica_case, supar=[area])
    assert by_area.converged
    by_pressure = _silica_solve(cea_module, silica_case, pi_p=[10.0 / by_area.P[-1]])
    assert by_pressure.converged
    for name in ("T", "enthalpy", "Isp_vacuum"):
        assert getattr(by_area, name)[-1] == pytest.approx(
            getattr(by_pressure, name)[-1], abs=3.0e-5
        )


def test_silica_missing_intermediate_phase_stops(cea_module, silica_case):
    reactants, products, weights = silica_case
    products = cea_module.Mixture([
        name for name in products.species_names if name != "SiO2(b-qz)"
    ])
    with pytest.warns(RuntimeWarning, match="CEA_NOT_CONVERGED"):
        solution = _silica_solve(cea_module, (reactants, products, weights), pi_p=[6000.0])
    assert not solution.converged
    assert solution.num_pts == 2
