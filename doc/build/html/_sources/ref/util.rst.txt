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

