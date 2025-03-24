import numpy as np

def add_arrow(line, position=None, direction='right', size=15, color=None):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if len(xdata) < 2:
        return

    if position is None:
        position = (xdata[0]+xdata[-1])/2  # find approximate middle of data
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
        xytext=(xdata[start_ind], ydata[start_ind]),
        xy=(xdata[end_ind], ydata[end_ind]),
        arrowprops=dict(arrowstyle="-|>", color=color),
        size=size
    )
    return


def complex_axes(ax, q, x_off=None, y_off=None, sz=None, grid=True, color="k"):
    # offset of x label
    if x_off is None:
        x_off = [0, 0]

    # offset of y label
    if y_off is None:
        y_off = [0, 0]

    # fontsize
    if sz is None:
        sz = 10

    ax.set_aspect('equal')              # Force x and y to have ratio 1:1
    ax.grid(grid, which='both')         # Enable grid
    ax.axvline(x=0, color=color)        # Plot y-axis
    ax.axhline(y=0, color=color)        # Plot x-xis

    x_min, x_max = ax.get_xlim()        # Get ranges for heuristic placement of text
    x_range = x_max - x_min
    if x_min < 0:
        x_min = 0

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    if y_min < 0:
        y_min = 0

    # Make y-label
    ax.text(x_min + x_range / 50 + y_off[0], y_max - y_range / 10 + y_off[1],
            r"$\mathrm{Im}\, " + f"{q}$",
            wrap=True, fontsize=sz)

    # Make x-label
    ax.text(x_max - x_range / 10 + x_off[0], y_min - y_range / 15 + x_off[1],
            r"$\mathrm{Re}\, " + f"{q}$",
            wrap=True, fontsize=sz)
    return
