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
