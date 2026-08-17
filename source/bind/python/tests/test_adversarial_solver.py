import warnings

import numpy as np


def _custom_ap_composite(cea_module, ions):
    components = [
        cea_module.Reactant(
            "ADV_AP",
            formula={"CL": 1, "H": 4, "N": 1, "O": 4},
            molecular_weight=117.48906,
            enthalpy=-295425.90124,
            enthalpy_units="J/mol",
            temperature=298.15,
        ),
        cea_module.Reactant(
            "ADV_HTPB",
            formula={"C": 200, "H": 302, "O": 2},
            molecular_weight=2738.53668,
            enthalpy=-4205218.478192,
            enthalpy_units="J/mol",
            temperature=298.15,
        ),
        cea_module.Reactant(
            "ADV_AL",
            formula={"AL": 1},
            molecular_weight=26.981538,
            enthalpy=0.0,
            enthalpy_units="J/mol",
            temperature=298.15,
        ),
    ]
    reactants = cea_module.Mixture(components, ions=ions)
    products = cea_module.Mixture(components, products_from_reactants=True, ions=ions)
    weights = np.array([0.70, 0.15, 0.15])
    hc = reactants.calc_property(cea_module.ENTHALPY, weights, 298.15) / cea_module.R
    return reactants, products, weights, hc


def _solve_composite(
    cea_module, ions, area_ratios=(5.0, 10.0, 20.0, 40.0), frozen=False
):
    reactants, products, weights, hc = _custom_ap_composite(cea_module, ions)
    solver = cea_module.RocketSolver(
        products, reactants=reactants, ions=ions, frozen_rephase=frozen
    )
    solution = cea_module.RocketSolution(solver)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        solver.solve(
            solution,
            weights,
            68.94757,
            supar=list(area_ratios),
            n_frz=1 if frozen else None,
            hc=hc,
        )
    return solution


def _assert_finite_rocket_solution(solution):
    for values in (
        solution.T,
        solution.P,
        solution.Mach,
        solution.ae_at,
        solution.Isp,
        solution.Isp_vacuum,
    ):
        assert np.all(np.isfinite(values))


def test_ap_composite_without_ions_attains_extreme_schedule(cea_module):
    solution = _solve_composite(cea_module, False)

    assert solution.converged
    assert solution.num_pts == 6
    np.testing.assert_allclose(solution.ae_at[2:], [5.0, 10.0, 20.0, 40.0], rtol=5.0e-3)
    _assert_finite_rocket_solution(solution)


def test_failed_ionized_composite_never_reports_success_or_stale_stations(cea_module):
    outcomes = []
    for _ in range(3):
        solution = _solve_composite(cea_module, True)
        outcomes.append((solution.converged, solution.num_pts))
        assert not solution.converged
        assert solution.num_pts < 6
        _assert_finite_rocket_solution(solution)

    assert outcomes == [outcomes[0]] * len(outcomes)


def test_frozen_rephased_composite_handles_dense_extreme_schedule(cea_module):
    schedule = (2, 3, 4, 5, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 100)
    solution = _solve_composite(cea_module, False, schedule, frozen=True)

    assert solution.converged
    assert solution.num_pts == 2 + len(schedule)
    np.testing.assert_allclose(solution.ae_at[2:], schedule, rtol=5.0e-3)
    _assert_finite_rocket_solution(solution)


def test_unordered_area_schedule_has_no_cross_station_state_leak(cea_module):
    schedule = (100.0, 2.0, 40.0, 5.0, 20.0)
    solution = _solve_composite(cea_module, False, schedule)

    assert solution.converged
    np.testing.assert_allclose(solution.ae_at[2:], schedule, rtol=5.0e-3)
    _assert_finite_rocket_solution(solution)
