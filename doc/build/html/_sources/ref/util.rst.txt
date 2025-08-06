.. module:: stoch_imp.util

============
Util
============

This module offers a bunch of helper functions and utility functions. These functions take into account the difference
between symbolic and numeric calculations, and are used in various calculations.

calc_curr_like
--------------
The probability current from a state :math:`m` to a state :math:`n` is given by

.. math::
    J_{mn} = W_{mn} P_{n} - W_{nm} P_{m}.

Due to the nature of the theory, these types of equations show up a lot in calculating the conductances. This function
calculates this expression for an arbitrary matrix and vector.

.. autofunction:: calc_curr_like

deep_simp
---------
This function simplifies symbolic objects in place by first simplifying numerically, then symbolically. If the object
passed is an instance of (a subclass of) :class:`~stoch_imp.core_classes.Mat`, the internal matrix is simplified.

.. autofunction:: deep_simp

gram_schmidt
------------
When orthogonalizing the vectors, a non-canonical inner product is used. This function implements the Gram-Schmidt
algorithm using the inner product given by

.. math::
    \langle v, w \rangle = \sum_i \frac{v_i w_i}{P^{eq}_i}

where :math:`P^{eq}` is given as optional input to the function. If no :math:`P^{eq}` is given a vector consisting
only of ones is used, yielding the standard inner product.

.. autofunction:: gram_schmidt

inner
-----

This is a simple helper function performing the inner product as defined by

.. math::
    \langle v, w \rangle = \sum_i \frac{v_i w_i}{P^{eq}_i}

where :math:`P^{eq}` is given as optional input to the function. If no :math:`P^{eq}` is given a vector consisting
only of ones is used, yielding the standard inner product.

.. autofunction:: inner