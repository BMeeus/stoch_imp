.. module:: stoch_imp.WMatrix

====================
WMatrix
====================
This class is the one that is mainly interacted with, and handles periodically driven transition matrices.
The periodic driving is modeled through a driving symbol, which must be provided at initialisation. If this is not
provided, an attempt is made to infer this from the input. The class provides methods for calculating transition
matrices at equilibrium, first-order driving matrices, computing conductivity coefficients, and evaluating
frequency-dependent conductivities.

Creating a WMatrix
------------------

To create a WMatrix, there are two options:

**Creating an empty matrix**
    By passing an integer :math:`n`, an empty :math:`(n,n)` transition matrix is created. You should also add the driving symbol
    :code:`ds` that represents the driving in the system. By default, the equilibrium value of the driving is 0,
    but this can be changed by using the :code:`eq` keyword at initialization. Below is sample code to create an empty
    :math:`(4, 4)` transition matrix.

    >>> from sympy import Symbol
    >>> from stoch_imp import WMatrix
    >>> n = 4
    >>> mu = Symbol("mu", real=True)
    >>> muEq = 1
    >>> w = WMatrix(n, ds=mu, eq=muEq)
    >>> print(w)
    ⎡0  0  0  0⎤
    ⎢          ⎥
    ⎢0  0  0  0⎥
    ⎢          ⎥
    ⎢0  0  0  0⎥
    ⎢          ⎥
    ⎣0  0  0  0⎦


**Initializing a prebuilt matrix**
    By passing an array (or another matrix-like object) you create a populated transition matrix. You should still
    define the driving symbol, but a naive attempt is made to infer the driving symbol from the array. Again the
    equilibrium value can be changed. Some basic checks are performed on the array to ensure that it is a valid
    transition matrix.

    >>> from stoch_imp import WMatrix
    >>> from sympy import Symbol
    >>> mu = Symbol("mu", real=True)
    >>> muEq = 1
    >>> w = WMatrix([[-mu, 2 * mu], [mu, -2 * mu]], eq=muEq)
    Driving symbol set to μ
    >>> print(w)
    ⎡-μ  2⋅μ ⎤
    ⎢        ⎥
    ⎣μ   -2⋅μ⎦
    >>> E = Symbol("E")
    >>> w = WMatrix([[-mu, 2 * mu * E], [mu, -2 * mu * E]], eq=muEq)
    ValueError: Ambiguity in driving symbol, please provide a specific symbol

Building the system
-------------------
You add transitions by using the :meth:`~stoch_imp.core_classes.TrMatrix.add_trans` method. By default the transition
matrix is 1-indexed. We can now add a transition from state :math:`i` to state :math:`j` with rate :math:`r` by using
:code:`.add_trans(i, j, r)`. The matrix gets updated, and both the transition :math:`i \to j` and :math:`j \to i` are
added. For example:

    >>> from sympy import Symbol, exp
    >>> from stoch_imp import WMatrix
    >>> mu = Symbol("mu", real=True)
    >>> w = WMatrix(3, ds=mu)
    >>> w.add_trans(1, 2, exp(mu))
    >>> print(w)
    ⎡  μ   -μ    ⎤
    ⎢-ℯ   ℯ     0⎥
    ⎢            ⎥
    ⎢ μ     -μ   ⎥
    ⎢ℯ    -ℯ    0⎥
    ⎢            ⎥
    ⎣ 0    0    0⎦

.. warning::
    It is generally not advisable to add transitions directly to the matrix using item assignment. Although this can be
    done, the matrix diagonal needs to be recalculated using :meth:`~stoch_imp.core_classes.TrMatrix.calc_diag`.
    The :meth:`~stoch_imp.core_classes.TrMatrix.add_trans` method on the other hand takes care of this for you, along
    with other quality of life improvements.

Calculating equilibrium matrix
------------------------------
Once the system is fully built, we can calculate the equilibrium matrix by calling :meth:`~WMatrix.calc_weq`. The result
is an :class:`~stoch_imp.EqMatrix.EqMatrix` that is stored in the :code:`weq` property. Once calculated, the
:meth:`~WMatrix.calc_weq` method returns the precalculated value. If a recalculation is needed, you can use the
:code:`force` flag.

By default the value stored in :code:`WMatrix.eq` is used, but a custom value can be passed on when calling
:meth:`~WMatrix.calc_weq`.

Calculating conductances
------------------------
The calculation of the conductances is the main focus of this package. They can be obtained by using the
:meth:`~WMatrix.get_conds` method. When no input is given with the method, a heuristic is used to calculate all
non-trivial conductances. If only certain transitions are needed, you can input them as a list of tuples. For example,
if the conductances :math:`1 \to 2`, :math:`1 \to 3` and :math:`2 \to 3` are needed, you use :code:`w.get_conds([(1, 2), (1, 3), (2, 3)])`.
The method returns a list containing the indices of the transition and the conduction as a callable function.
An example of how to use this in practice:


.. code-block::

    with si.create_fig(1) as (fig, ax):

        # Define range of frequencies
        om_arr = np.logspace(-10, 2, 10000)

        for ind, cond in w.get_conds([(1, 2), (3, 4), (1, 3)]):
            res_arr = cond(om_arr)
            # plot result
            line = ax.plot(np.real(res_arr), np.imag(res_arr), label="${}\\to{}$".format(*ind))[0]
            si.add_arrow(line)

        # Add axes
        si.complex_axes(ax, r"\sigma_{mn}(\omega)")

        ax.set_title("Periodically driven current", fontname="serif", fontsize=18)
        ax.legend()

For more info on the plotting of conductances, refer to :doc:`Pyplot`

Numeric calculations
--------------------

By default, the WMatrix is treated as a symbolic matrix. If numeric calculations are preferred, use the
:meth:`~WMatrix.to_num` to convert the matrix and eigensystem to numeric types. This method does not calculate anything,
it merely converts all stored quantities to numeric and changes the type of the matrix. Any future calculations will be done numerically. To
convert back to symbolical calculations, use :meth:`~WMatrix.to_sym`. Since WMatrix objects are by definition symbolic
(they contain the driving symbol), :meth:`~WMatrix.to_num` only converts the stored matrices to numeric and changes the
:code:`is_symbolic` flag to False.

Reference
---------
.. autoclass:: WMatrix
    :members:
    :inherited-members:
