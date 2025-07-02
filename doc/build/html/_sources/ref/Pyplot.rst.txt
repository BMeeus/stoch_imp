.. module:: stoch_imp.pyplot_funcs

======================
Pyplot Functionalities
======================

This module contains functions that help with plotting the conductances (among other things). A quick overview:

**create_fig()**
    is a context manager used to deal with subplots, simplifying the math and tick properties and ensuring consistent
    formatting and layout. It can be customised to the specific usecase.

**add_arrow()**
    is a function that adds an arrow onto a line object.

**complex_axes()**
    is a function that adds complex axes with appropriate labels to the plot. Placing of the labels can be fine-tuned.


.. autofunction:: create_fig

.. autofunction:: add_arrow

.. autofunction:: complex_axes
