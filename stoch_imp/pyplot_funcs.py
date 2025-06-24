from matplotlib.lines import Line2D
from matplotlib.pyplot import Axes, rcParams, subplots, tight_layout, savefig, show
from numpy import (absolute, argmin, array)
from contextlib import contextmanager
from typing import Union, Optional


@contextmanager
def create_fig(subplt: Union[int, tuple[int, int]],
               save_path: Optional[str] = None,
               dpi: Optional[int] = 600,
               tight: Optional[bool] = True,
               tick_dict: Optional[dict] = None,
               **kwargs):
    """
    Context manager to create and configure a matplotlib figure with subplot(s).

    This function initializes a matplotlib figure and subplots according to the
    given layout, applies custom tick parameters, and manages figure display and
    optional saving upon exit. It also configures LaTeX rendering and font settings
    for consistent styling.

    :param subplt: Number of subplots (if int) or subplot grid shape as (rows, cols)
    :type subplt: int or tuple[int, int]
    :param save_path: If specified, the figure is saved to this file path on exit
    :type save_path: str or None
    :param dpi: Resolution in dots per inch for saved figure (default: 600)
    :type dpi: int or None
    :param tight: Whether to apply `tight_layout()` to minimize padding (default: True)
    :type tight: bool or None
    :param tick_dict: Dictionary of parameters passed to `tick_params()` (default sets serif font)
    :type tick_dict: dict or None
    :param kwargs: Additional keyword arguments passed to `matplotlib.pyplot.subplots()`

    :yield: Tuple containing the figure and axes objects
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """

    if tick_dict is None:
        tick_dict = {'labelfontfamily':'serif'}

    rcParams.update({
        "text.usetex": True,
        "font.family": "mathpazo"
    })

    if isinstance(subplt, int):
        fig, ax = subplots(1, subplt, **kwargs)
    elif isinstance(subplt, tuple) or isinstance(subplt, list):
        fig, ax = subplots(subplt[0], subplt[1], **kwargs)
    else:
        raise TypeError(f"subplt argument must be int or [int, int], is {type(subplt)}")

    try:
        for axis in ax:
            axis.tick_params(**tick_dict)
    except TypeError:
        ax.tick_params(**tick_dict)
    yield fig, ax

    if tight:
        tight_layout()

    if save_path:
        savefig(save_path, dpi=dpi)
    show()


def add_arrow(line: Line2D,
              position: float = None,
              direction: str = 'right',
              size: int = 15,
              color: str = None) -> None:
    """
    add an arrow to a line.

    :param line: The Line object to which the arrow is added
    :type line: Line2D
    :param position: x-position of the arrow. If None, mean of xdata is taken
    :type position: float
    :param direction: 'left' or 'right'
    :type direction: str
    :param size: size of the arrow in fontsize points
    :type size: int
    :param color: if None, line color is taken.
    :type color: str
    """
    if color is None:
        color = line.get_color()

    xdata = array(line.get_xdata())
    ydata = array(line.get_ydata())

    if len(xdata) < 2:
        return

    if position is None:
        position = (xdata[0] + xdata[-1]) / 2  # find approximate middle of data
    # find the closest index
    start_ind = argmin(absolute(xdata - position))
    if direction in ('right', 'r', '>', '->'):
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    # Add arrow note with empty string
    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle="-|>", color=color),
                       size=size
                       )
    return



def complex_axes(ax: Axes,
                 q: str,
                 r_off: list[float] | tuple[float, float] = (0, 0),
                 i_off: list[float] | tuple[float, float] = (0, 0),
                 sz: int = 10,
                 grid: bool = True,
                 color: str = "k") -> None:
    """
    Add complex axes to a figure, with labels of imaginary and real parts. These labels are heuristically placed and can
    be offset if necessary.

    The offset uses axis units. For example: if the label appears at (0.2, 5.4) on the axis
    but should be placed at (0.3, 5.1), the offset should be given as (0.1, -0.3)

    :param ax: An axes object
    :type ax: Axes
    :param q: The quantity plotted. This can be a string containing latex syntax.
    :type q: str
    :param r_off: Offset of the real axis label, given in axis units.
    :type r_off: (float, float)
    :param i_off: Offset of the imaginary axis label, given in axis units.
    :type i_off: (float, float)
    :param sz: Font size of the labels
    :type sz: int
    :param grid: If True, adds a grid to the axes object.
    :type grid: bool
    :param color: If None, black is used.
    :type color: str
    :return: None
    """

    ax.set_aspect('equal')  # Force x and y to have ratio 1:1
    ax.grid(grid, which='both')  # Enable grid
    ax.axvline(x=0, color=color)  # Plot y-axis
    ax.axhline(y=0, color=color)  # Plot x-xis

    x_min, x_max = ax.get_xlim()  # Get ranges for heuristic placement of text
    x_range = x_max - x_min
    if x_min < 0 < x_max:
        x_min = 0

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    if y_min < 0 < y_max:
        y_min = 0

    # Make y-label
    ax.text(x_min + x_range / 50 + i_off[0], y_max - y_range / 10 + i_off[1],
            r"$\mathrm{Im}\, " + f"{q}$",
            wrap=True, fontsize=sz)

    # Make x-label
    ax.text(x_max - x_range / 10 + r_off[0], y_min + y_range / 50 + r_off[1],
            r"$\mathrm{Re}\, " + f"{q}$",
            wrap=True, fontsize=sz)
    return
