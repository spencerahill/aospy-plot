"""Classes for creating multi-panel figures using data generated via aospy."""
from aospy.calc import Calc, CalcInterface
from aospy.io import to_dup_list
from aospy.operator import Operator
from aospy.utils import to_hpa
import numpy as np
import matplotlib.pyplot as plt

fig_specs = (
    'fig_title', 'n_row', 'n_col', 'row_size', 'col_size', 'n_ax',
    'subplot_lims', 'cbar_ax_lim', 'cbar_kwargs', 'cbar_ticks',
    'cbar_ticklabels', 'cbar_label', 'cbar_label_kwargs',
    'cbar_left_label', 'cbar_left_label_coords', 'cbar_left_label_kwargs',
    'cbar_right_label', 'cbar_right_label_coords', 'cbar_right_label_kwargs',
    'verbose'
)

class Fig(object):
    """Class for producing figures with one or more panels."""
    def __init__(self, fig_params, n_ax=1, n_plot=1, n_data=1, n_row=1,
                 n_col=1, date_range=None, intvl_in=None, intvl_out=None,
                 dtype_in_time=None, dtype_in_vert=None, dtype_out_time=None,
                 dtype_out_vert=None, level=None, **kwargs):
        self.__dict__ = vars(fig_params)

        self.n_ax = n_ax
        self.n_plot = n_plot
        self.n_data = n_data
        self.n_row = n_row
        self.n_col = n_col
        self._set_n_ax_plot_data()

        self.date_range = date_range
        self.intvl_in = intvl_in
        self.intvl_out = intvl_out
        self.dtype_in_time = dtype_in_time
        self.dtype_in_vert = dtype_in_vert
        self.dtype_out_time = dtype_out_time
        self.dtype_out_vert = dtype_out_vert
        self.level = level

        self.ax = []

        # Accept all other keyword arguments passed in as attrs.
        for key, val in kwargs.items():
            setattr(self, key, val)

        self._expand_attrs_over_tree()
        self._make_ax_objs()

    def _set_n_ax_plot_data(self):
        """Set the number of axes, plots, and data."""
        # Set the number of Axes.
        if self.n_ax == 'all':
            self.n_ax = self.n_row*self.n_col
        else:
            assert self.n_ax <= self.n_row*self.n_col
        # Distribute n_plot across the Axes.
        self.n_plot = to_dup_list(self.n_plot, self.n_ax)
        # Distribute n_data across the Plots.
        self.n_data = to_dup_list(self.n_data, self.n_ax)
        for i, ndt in enumerate(self.n_data):
            self.n_data[i] = to_dup_list(ndt, self.n_plot[i])

    def _traverse_child_tree(self, value, level='data'):
        """
        Traverse the "tree" of child Ax, Plot, and Data objects by creating a
        nested list of three levels, each level cascading one child object
        class down (e.g. from Ax to Plot to Data).
        """
        assert level in ('fig', 'ax', 'plot', 'data')
        if level in ('ax', 'plot', 'data'):
            value = to_dup_list(value, self.n_ax)
            if level in ('plot', 'data'):
                for i, vi in enumerate(value):
                    value[i] = to_dup_list(vi, self.n_plot[i])
                    if level == 'data':
                        for j, vij in enumerate(value[i]):
                            value[i][j] = to_dup_list(
                                vij, self.n_data[i][j], single_to_list=False
                            )
        return value

    def _expand_specs(self, specs, spec_name):
        for attr in specs:
            value = getattr(self, attr, False)
            setattr(self, attr, self._traverse_child_tree(value, spec_name))

    def _expand_attrs_over_tree(self):
        """
        Replicate attrs such that each Ax-level attr is a list with length
        equal to the number of Axes.  Similarly, Plot-level attrs become
        two-level lists, the first level being corresponding to the Axes and
        the second level to the Plots within each Ax.  And likewise for
        Data-level attrs.
        """
        for specs, name in ((fig_specs, 'fig'),
                            (ax_specs, 'ax'),
                            (plot_specs, 'plot'),
                            (data_specs, 'data')):
            self._expand_specs(specs, name)

    def _locate_ax(self, panel_num):
        """Determine if the panel is interior, left, bottom-left, or bottom."""
        if panel_num == 0 or panel_num == self.n_col*(self.n_row - 1):
            return 'bottomleft'
        elif panel_num > self.n_col*(self.n_row - 1):
            return 'bottom'
        elif not panel_num % self.n_col:
            return 'left'
        else:
            return 'interior'

    def _make_ax_objs(self):
        """Create the Ax obj for each panel."""
        self.ax = [Ax(self, n, self._locate_ax(n))
                   for n in range(self.n_ax)]

    @staticmethod
    def __add_text(ax, coords, string, kwargs):
        ax.text(coords[0], coords[1], string, **kwargs)

    def _make_colorbar(self, ax):
        """Create colorbar for multi panel plots."""
        # Don't make if already made.
        if hasattr(self, 'cbar'):
            return
        self.cbar_ax = self.fig.add_axes(self.cbar_ax_lim)
        kwargs = self.cbar_kwargs if self.cbar_kwargs else dict()
        self.cbar = self.fig.colorbar(ax.Plot[0].handle, cax=self.cbar_ax,
                                      **kwargs)
        # Set tick properties.
        if np.any(self.cbar_ticks):
            self.cbar.set_ticks(self.cbar_ticks)
        if self.cbar_ticklabels not in (None, False):
            self.cbar.set_ticklabels(self.cbar_ticklabels)
        self.cbar.ax.tick_params(labelsize='x-small')
        # Add center, left, and right labels as desired.
        if self.cbar_label:
            var = self.var[0][0][0]
            if self.cbar_label == 'units':
                if self.dtype_out_vert[0][0][0] == 'vert_int':
                    label = var.units.vert_int_plot_units
                else:
                    label = var.units.plot_units
            else:
                label = self.cbar_label
            self.cbar.set_label(label, **self.cbar_label_kwargs)

        def make_cbar_label_args(obj, string):
            return [obj.cbar_ax] + [
                    getattr(obj, 'cbar_' + string + '_label' + suffix, False)
                    for suffix in ('_coords', '', '_kwargs')
            ]

        if self.cbar_left_label:
            self.__add_text(*make_cbar_label_args(self, 'left'))
        if self.cbar_right_label:
            self.__add_text(*make_cbar_label_args(self, 'right'))

    def create_fig(self):
        """Create the figure and set up the subplots."""
        self.fig = plt.figure(figsize=(self.n_col*self.col_size,
                                       self.n_row*self.row_size))
        self.fig.subplots_adjust(**self.subplot_lims)
        if self.fig_title:
            self.fig.suptitle(self.fig_title, fontsize=12)

        for n in range(self.n_ax):
            self.ax[n].ax = self.fig.add_subplot(self.n_row, self.n_col, n+1)

    def make_plots(self):
        """Render the plots in every Ax."""
        for n in range(self.n_ax):
            self.ax[n].make_plots()

    def savefig(self, *args, **kwargs):
        """Save the Fig using matplotlib's built-in 'savefig' method."""
        self.fig.savefig(*args, **kwargs)

    def draw(self):
        """Call the matplotlib method canvas.draw() to re-render the figure."""
        self.fig.canvas.draw()
