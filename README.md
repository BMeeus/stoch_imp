# Stochastic Impedance

## Overview
These files can be used to calculate the conductances for arbitrary systems, The general workflow is as follows:
- Define the transfer matrix $W$:
This can be done directly, or by remapping a matrix describing the transitions of the sites. To do this, use sympy to define the matrix as a function of the driving parameter.
- Calculate $W_{eq}$ and $W_1$:
By leveraging sympy methods, $W_{eq}$ and $W_1$ can be calculated for a wide range of different couplings, dependencies and driving parameters. These matrices are then saved as numpy arrays.
- Calculate the coefficients:
From the above matrices, the coefficients of the eigenvector expansion of the conductance are calculated. This is done in a seperate file that is system agnostic, and so can be used as a black box calculation. The coefficients along with other relevant results are saved as numpy arrays.
- Calculate and plot the conductances:
From the coefficients, the conductances can be calculated. These can then be analysed, plotted, or compared as needed. The calculation of the conductances is _not_ included in the coefficient calculation for ease of use.

## Files
The files are split in _general/template files_ and _work files_.

**General/template files** are those files which should in essence not be changed. They are either central to the woking of the package or serve as templates for use of the package. These include:
### calculate_coeffs.py
The central program that calculates the coefficients starting from any $Weq$ and $W1$. This can be used as black box and should never be changed without good reason.
### plot_conds.py
This is the template file to plot the conductances. It can easily be copied and adapted to fit specific systems and simulations.
### pyplot_funcs.py
This includes the functionalities used to plot the conductances. It contains two functions:
- add_arrow: This adds an arrow in the direction of increasing frequency.
- complex_axes: This heuristically adds complex axes together with axis labels to the figure.

For more information see the documentation.
### template*.py
These files are template files to calculate the $W$ matrix for different systems. They can be copied and modified as needed.

**Work files** are files connected to a specific system of simulation. They are less neat and commented. These are the files where the actual simulations are done. Many are depricated due to updates in the workflow.

## User manual
Below is a step-by-step guide to mounting a simulation and plotting the results. Following this one will be able to plot the statistic impedances of an arbitrary finite master equation system. This is purely a practical guide, for a mathematical explanation please consider stoch_imp_theory.pdf

1. Define the transfer matrices $W_{eq}$ and $\bar{W}$ (called `w1` in the code). In the code this is done through defining $W$ and deriving $W_{eq}$ and $\bar{W}$ from there. It is important that this is done _as on paper_: the matrix element $W_{ij}$ corresponds to `w[i, j]`.
The matrix is built using SymPy so that the derivation is easily implemented. It is not generally necessary to use SymPy or even define $W$ at all as long as the correct $W_{eq}$ and $\bar{W}$ are defined with the correct indexing. 
> IMPORTANT: Due to the fact that python uses column first numbering, this means that the matrix comes out transposed when printed. This is accounted for in the code.
2. Calculate the coefficients $A^k_{mn}$ by importing calculate_coeffs. The function takes $W_{eq}$ and $\bar{W}$ as arguments and returns $A^k_{mn}$, the eigenvalues $\lambda_k$ and the eigen vectors $v_k$. If the optional argument `pathname` is given, the function saves the results including $W_{eq}$ and $\bar{W}$ to a npz file with name and location specified in the path.
3. Define the transitions for which the conductances are computed. Generally one wants to calculate the conductance for only _some_ $m, n$, because the transition does not exist, because of symmetry reasons, etc. It is then important to define which currents are taken into consideration. This can either be done by analysing $W_{eq}$ and $\bar{W}$ to find non-trivial transitions, or manually by specifying the exact transitions of interest.
A transition $i \to j$ is represented in the code as `[i-1, j-1]`. Be aware of the zero indexing! 
4. Calculate the conductances: this is best done by a separate function as can be found in `plot_conds.py`. Because the coefficients are given by a NumPy array, the calculation can be done for an entire array of inputs at once.
> IMPORTANT: The indexing of the coefficients is such that the transition $i \to j$ corresponds to the `coeffs[i-1, j-1, :]`. This is partly due to zero indexing and partly by design for ease of use.
