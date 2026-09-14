#include <math.h>
#include <stdio.h>
#include "cea.h"

#define CHECK(expr) do { if (!(expr)) { fprintf(stderr, "Failed at line %d\n", __LINE__); return 1; } } while (0)

int main(void)
{
    const cea_string reactants[] = {"N2", "SiO2(L)"};
    const cea_string products[] = {"N2", "SiO2(a-qz)", "SiO2(b-qz)", "SiO2(b-crt)", "SiO2(L)"};
    const cea_real weights[] = {0.5, 0.5}, ratios[] = {6000.0};
    cea_mixture reac, prod;
    cea_rocket_solver solver;
    cea_rocket_solution soln;
    cea_real temperature[3];
    cea_int npts;
    cea_solver_opts opts;

    CHECK(cea_rocket_solver_set_frozen_rephase(NULL, true) == CEA_INVALID_INDEX);
    CHECK(cea_init() == CEA_SUCCESS);
    CHECK(cea_mixture_create(&reac, 2, reactants) == CEA_SUCCESS);
    CHECK(cea_mixture_create(&prod, 5, products) == CEA_SUCCESS);
    CHECK(cea_solver_opts_init(&opts) == CEA_SUCCESS);
    opts.reactants = reac;
    CHECK(cea_rocket_solver_create_with_options(&solver, prod, opts) == CEA_SUCCESS);
    CHECK(cea_rocket_solution_create(&soln, solver) == CEA_SUCCESS);

    /* Creation defaults to disabled; enabling and disabling affect later solves. */
    for (int mode = 0; mode < 3; ++mode) {
        if (mode != 0) CHECK(cea_rocket_solver_set_frozen_rephase(solver, mode == 1) == CEA_SUCCESS);
        cea_err status = cea_rocket_solver_solve_iac(solver, soln, weights, 10.0,
            ratios, 1, NULL, 0, NULL, 0, -1, 2200.0, false, 0.0, false);
        CHECK(status == (mode == 1 ? CEA_SUCCESS : CEA_NOT_CONVERGED));
        CHECK(cea_rocket_solution_get_size(soln, &npts) == CEA_SUCCESS);
        CHECK(npts == (mode == 1 ? 3 : 2));
        if (mode == 1) {
            CHECK(cea_rocket_solution_get_property(soln, CEA_ROCKET_TEMPERATURE, npts, temperature) == CEA_SUCCESS);
            CHECK(isfinite(temperature[2]) && temperature[2] > 700.0 && temperature[2] < 850.0);
        }
    }
    CHECK(cea_rocket_solution_destroy(&soln) == CEA_SUCCESS);
    CHECK(cea_rocket_solver_destroy(&solver) == CEA_SUCCESS);
    CHECK(cea_mixture_destroy(&prod) == CEA_SUCCESS);
    CHECK(cea_mixture_destroy(&reac) == CEA_SUCCESS);
    return 0;
}
