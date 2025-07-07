.. module:: stoch_imp.EqMatrix

====================
EqMatrix
====================
This module introduces the EqMatrix class, a class handling all equilibrium matrices and related calculations. It
provides methods to calculate the eigensystem and stable solution, as well as a simple checker for detailed balance.

Creating an equilibrium matrix
------------------------------

To create an equilibrium matrix, we declare an EqMartix object. As input one can use either an array like object or
an integer. If an integer `n` is used, EqMatrix will be an empty `(n, n)` matrix:

    >>> from stoch_imp import EqMatrix
    >>> weq = EqMatrix(3)
    >>> print(weq)
    [0 0 0]
    [0 0 0]
    [0 0 0]
    >>> EqMatrix([[-0.5, 1], [0.5, -1]])
    [-0.5  1]
    [ 0.5 -1]

When creating the equilibrium matrix that is not a valid transfer matrix (non-zero column sums or negative rates) an
error will be raised.

Building the system
-------------------

Starting from an empty matrix, interactions can be added by using the :meth:`~EqMatrix.add_trans` method:

    >>> from stoch_imp import EqMatrix
    >>> weq = EqMatrix(3)
    >>> weq.add_trans(1, 2, r=2, ri=0.5)
    [-2  0.5 0]
    [ 2 -0.5 0]
    [ 0  0   0]

.. caution::
    It is generally not advisable to add transitions directly to the matrix using item assignment. Although this can be
    done, the matrix diagonal needs to be recalculated using :meth:`~stoch_imp.core_classes.TrMatrix.calc_diag`.
    The :meth:`~stoch_imp.core_classes.TrMatrix.add_trans` method on the other hand takes care of this for you.

Calculating eigensystem
-----------------------

After building the system, we can calculate the eigensystem and equilibrium distribution using :meth:`~EqMatrix.calc_eig`. These
values are then returned and stored as attributes of the object. By default, detailed balance is checked.

.. note::
    The method :meth:`~EqMatrix.calc_peq` is basically an alias for :meth:`~EqMatrix.calc_eig`. Because the equilibrium distribution is
    calculated using the eigensystem, the full calculation of the eigensystem is performed to avoid wasting resources.

If the eigensystem is already calculated and a recalculation is needed, you *must* use the `force` flag. Otherwise the
old values are returned.

Numeric calculations
--------------------

By default, the EqMatrix is treated as a symbolic matrix. If numeric calculations are preferred, use the :meth:`~EqMatrix.to_num()`
to convert the matrix and eigensystem to numeric types. This method does not calculate anything, it merely converts all
stored quantities to numeric and changes the type of the matrix. Any future calculations will be done numerically. To
convert back to symbolical calculations, use :meth:`~EqMatrix.to_sym`.

Reference
---------
.. autoclass:: EqMatrix
    :members: