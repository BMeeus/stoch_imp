.. module:: stoch_imp.WMatrix

====================
WMatrix
====================
This class is the one that is mainly interacted with, and handles periodically driven transition matrices.
The periodic driving is modeled through a driving symbol, which must be provided at initialisation. If this is not
provided, an attempt is made to infer this from the input. The class provides methods for calculating transition
matrices at equilibrium, first-order driving matrices, computing conductivity coefficients, and evaluating
frequency-dependent conductivities.

.. autoclass:: WMatrix
    :members:
    :inherited-members:
