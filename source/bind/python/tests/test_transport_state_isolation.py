import numpy as np


def test_transport_does_not_change_frozen_rocket_thermodynamics(cea_module):
    reactants = cea_module.Mixture(["CH4", "O2"])
    products = cea_module.Mixture(["CH4", "O2"], products_from_reactants=True)
    weights = reactants.moles_to_weights(np.array([1.0, 2.0]))
    hc = reactants.calc_property(cea_module.ENTHALPY, weights, 298.15) / cea_module.R
    solutions = []
    for transport in (False, True):
        solver = cea_module.RocketSolver(
            products, reactants=reactants, transport=transport
        )
        solution = cea_module.RocketSolution(solver)
        solver.solve(
            solution,
            weights,
            68.94757,
            supar=[2.0, 5.0, 10.0, 40.0, 100.0],
            n_frz=1,
            hc=hc,
        )
        assert solution.converged
        solutions.append(solution)

    np.testing.assert_allclose(solutions[1].T, solutions[0].T, rtol=1.0e-10)
    np.testing.assert_allclose(solutions[1].P, solutions[0].P, rtol=1.0e-10)
    np.testing.assert_allclose(
        solutions[1].Isp_vacuum, solutions[0].Isp_vacuum, rtol=1.0e-10
    )
