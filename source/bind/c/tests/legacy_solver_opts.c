/* Frozen subset of the pre-rephasing C declarations (v3.3.4).
 * Deliberately do not include cea.h: recompiling against a changed layout
 * would hide the binary-compatibility regression this client detects. */
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include "cea_enum.h"

typedef enum { CEA_ERROR_CODE_ENUM } cea_err;
typedef enum { CEA_EQUILIBRIUM_SIZE_ENUM } cea_equilibrium_size;
typedef struct cea_mixture_t *cea_mixture;
typedef struct cea_rocket_solver *cea_rocket_solver;
typedef const char *cea_string;
typedef struct {
    double trace;
    bool ions;
    bool transport;
    cea_mixture reactants;
    int ninsert;
    const cea_string *insert;
    bool smooth_truncation;
    double truncation_width;
} cea_solver_opts;

cea_err cea_init(void);
cea_err cea_solver_opts_init(cea_solver_opts *opts);
cea_err cea_mixture_create(cea_mixture *mixture, int n, const cea_string names[]);
cea_err cea_mixture_destroy(cea_mixture *mixture);
cea_err cea_rocket_solver_create_with_options(cea_rocket_solver *solver, cea_mixture products, cea_solver_opts opts);
cea_err cea_rocket_solver_get_size(cea_rocket_solver solver, cea_equilibrium_size variable, int *value);
cea_err cea_rocket_solver_destroy(cea_rocket_solver *solver);

#define CHECK(expr) do { if (!(expr)) { fprintf(stderr, "Failed at line %d\n", __LINE__); return 1; } } while (0)

int main(void)
{
    struct { cea_solver_opts opts; unsigned char sentinel[16]; } probe;
    const cea_string names[] = {"N2"};
    cea_mixture mixture;
    cea_rocket_solver solver;
    int n;
    memset(&probe, 0xa5, sizeof(probe));
    CHECK(cea_solver_opts_init(&probe.opts) == CEA_SUCCESS);
    for (unsigned int i = 0; i < sizeof(probe.sentinel); ++i) CHECK(probe.sentinel[i] == 0xa5);
    CHECK(cea_init() == CEA_SUCCESS);
    CHECK(cea_mixture_create(&mixture, 1, names) == CEA_SUCCESS);
    probe.opts.reactants = mixture;
    CHECK(cea_rocket_solver_create_with_options(&solver, mixture, probe.opts) == CEA_SUCCESS);
    CHECK(cea_rocket_solver_get_size(solver, CEA_NUM_REACTANTS, &n) == CEA_SUCCESS && n == 1);
    CHECK(cea_rocket_solver_destroy(&solver) == CEA_SUCCESS);
    CHECK(cea_mixture_destroy(&mixture) == CEA_SUCCESS);
    return 0;
}
