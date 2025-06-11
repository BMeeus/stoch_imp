from matplotlib.lines import Line2D
from matplotlib.pyplot import Axes
from numpy import (absolute, argmin, array)


def add_arrow(line: Line2D,
              position: float = None,
              direction: str = 'right',
              size: int = 15,
              color: str = None) -> None:
    """
    add an arrow to a line.

    :param line: (Line2D) The Line object to which the arrow is added
    :param position: (float)  x-position of the arrow. If None, mean of xdata is taken
    :param direction: (str) 'left' or 'right'
    :param size: (int) size of the arrow in fontsize points
    :param color: (str) if None, line color is taken.
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

    :param ax: (Axes) An axes object
    :param q: (str) The quantity plotted. This can be a string containing latex syntax.
    :param r_off: ( (float, float) ) Offset of the real axis label, given in axis units.
    :param i_off: ( (float, float) ) Offset of the imaginary axis label, given in axis units.
    :param sz:  (Int) Font size of the labels
    :param grid: (bool) If True, adds a grid to the axes object.
    :param color: (str) If None, black is used.
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
