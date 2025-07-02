.. Stochastic Impedance documentation master file, created by
   sphinx-quickstart on Tue Jun 24 13:52:50 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the StochImp docs!
=============================

This package implements an algorithm to calculate the stochastic impedance between states of an arbitrary periodically
driven finite state stochastic system fulfilling detailed balance. It includes several classes for handling  the $W$
matrix and related transition matrices. It also includes some functionality to ease plotting with MatPlotLib.

The algorithms default to symbolic calculations, but if necessary all calculations can be done numerically.

.. note::

   This project is under active development.

Installation
------------
There is currently no installer implemented for the package. The best way to install is to download the package files or
fork the project. Once done, the package can be imported as usual.


Features
--------
- Define transition matrices for stochastic systems
- Calculate eigenspaces and equilibrium distributions
- Calculate stochastic impedances of transitions between sites
- Simplify plotting of conductances
- Supports both symbolic and numeric calculations.


Contents
--------
.. toctree::
   :maxdepth: 3

   tutorial/tutorial
   howto/howto
   ref/reference
   explanation/explanation

License
-------

This project does not currently fall under a license.
