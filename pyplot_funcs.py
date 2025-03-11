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

    if position is None:
        position = (xdata[0]+xdata[-1])/2
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


def complex_axes(ax, q, x_off=None, y_off=None):
    if x_off is None:
        x_off = [0, 0]

    if y_off is None:
        y_off = [0, 0]

    ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax.axvline(x=0, color='k')
    ax.axhline(y=0, color='k')
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.text(x_range / 50 + y_off[0], y_max - y_range / 10 + y_off[1], r"$\mathrm{Im}\, " + f"{q}$", wrap=True)
    ax.text(x_max - x_range / 10 + x_off[0], -y_range / 15 + x_off[1], r"$\mathrm{Re}\, " + f"{q}$", wrap=True)
    return