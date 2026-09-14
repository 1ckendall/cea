Python
******

The Python API provides a convenient way to interact with the program through Python, while maintaining the performance of the underlying Fortran code.
No unit conversions are performed by default; inputs and outputs are in the documented CEA units, and users are responsible for converting as needed.
Use the conversion factors in :mod:`cea.units` when working across unit systems.

Mixture
-------
The :class:`~cea.Mixture` class is used to define a mixture of product or reactant species. It allows the user to specify the composition of the mixture and provides methods to compute thermodynamic curve fit properties.
The instances of this class are then passed as inputs to the available solver classes (e.g., :class:`~cea.EqSolver`, :class:`~cea.RocketSolver`, :class:`~cea.ShockSolver`, or :class:`~cea.DetonationSolver`).
Custom reactants can be provided through :class:`~cea.Reactant` objects (including mixed lists of strings and Reactant objects).
For :class:`~cea.Reactant`, ``temperature`` is in K, and ``enthalpy`` must be paired with explicit
``enthalpy_units`` (for example ``"J/kg"`` or ``"kJ/mol"``). Weight-based enthalpy units require
``molecular_weight`` for conversion, while molar enthalpy units do not.

.. autoclass:: cea.Reactant
   :members:

.. autoclass:: cea.Mixture
   :members:

EqSolver
--------

.. autoclass:: cea.EqSolver
   :members:

EqSolution
----------

.. autoclass:: cea.EqSolution
   :members:

RocketSolver
------------

Pass ``reactants=reac`` to ``cea.RocketSolver(prod, reactants=reac)`` when the
reactant and product mixtures differ. If ``reactants`` is omitted, the solver
uses ``prod`` as both mixtures. In that case, ``solve`` requires one weight for
every species in ``prod.species_names`` order, including zeros for absent species.
The weights must be a one-dimensional array of length ``solver.num_reactants``;
an invalid shape or length raises ``ValueError`` before the native solver runs.

.. autoclass:: cea.RocketSolver
   :members:

``RocketSolver(..., frozen_rephase=True)`` permits a populated condensed species
to hand its fixed amount to a temperature-valid phase with the same elemental
stoichiometry during frozen expansion. Gas amounts and the total amount of each
condensed formula remain frozen. The replacement phase receives constant
enthalpy and entropy reference corrections at the handoff boundary so the
station solution remains continuous. This model excludes latent heat and full
two-phase equilibrium. Handoffs follow a fixed sequence of shared fit boundaries
from the freeze state, independent of temperature guesses. The current phase is
retained while its fit remains valid; at its boundary, the valid continuing phase
with the lowest database Gibbs energy is selected (exact ties use species names).
Missing intermediate phases are not skipped: the terminal phase retains the
50 K guard band, after which the solver returns a partial result. The option
defaults to ``False`` so existing frozen calculations are unchanged.

RocketSolution
--------------

.. autoclass:: cea.RocketSolution
   :members:

ShockSolver
-----------

Shock solves can emit a ``RuntimeWarning`` with ``LAST_VALID_SOLUTION`` when the
incident-equilibrium path retains a last valid state without converging the
shock iteration. In that case ``ShockSolution.last_error`` records the soft
failure, while ``ShockSolution.converged`` remains the authoritative strict
convergence signal.

.. autoclass:: cea.ShockSolver
   :members:

ShockSolution
-------------

.. autoclass:: cea.ShockSolution
   :members:

DetonationSolver
----------------

.. autoclass:: cea.DetonationSolver
   :members:

DetonationSolution
------------------

.. autoclass:: cea.DetonationSolution
   :members:
