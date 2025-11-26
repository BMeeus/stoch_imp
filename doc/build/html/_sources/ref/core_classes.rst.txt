.. module:: stoch_imp.core_classes

============
Core Classes
============

This module defines the core classes behind the algorithms. These will often not be used directly,
but are used under the hood.

Mat
---

This class handles all matrix like objects at the top level. It defines the basic properties and is used as the
foundation of all lower matrix classes.

.. autoclass:: Mat
    :members:
    :inherited-members:

TrMatrix
--------

This class handles all transition matrices, and safeguards the mathematical properties.
A transition matrix is a matrix :math:`W` whose off diagonal elements are non-negative, and whose diagonal elements are given by
:math:`- \sum_{i: i \neq j} W_{ij}`, such that the columns sum to zero.

.. autoclass:: TrMatrix
    :members:
    :inherited-members:

CoeffArray
----------

This class is a custom class for 3D arrays used to store the :math:`A^{(k)}_{mn}`
coefficients and simplify retrieval when calculating conductances.

.. autoclass:: CoeffArray
    :members:
    :inherited-members:

It forwards all indexing to numpy, and can be used as a numpy array. The main reason for its existence is that
it prints any symbolic expressions pretty.

ConstantObject
--------------
This is a mixin class used for explicitly constant objects, objects that do not contain any driving.
It allows the definition of a numeric matrix which is impossible for driven objects, as they would lose their dependence
on the driving symbol.

.. autoclass:: ConstantObject
    :members:

ConstantMatrix
--------------
This class is the most general class for constant matrices. Most constant objects that are not an :class:`EqMatrix`
will be of this type.

.. autoclass:: ConstantMatrix
    :members:

Functions
---------

.. autofunction:: arr_to_mat

.. autofunction:: check_diag

.. autofunction:: check_rates