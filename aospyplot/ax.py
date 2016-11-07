"""Classes for creating multi-panel figures using data generated via aospy."""
import logging
import string

from aospy.io import to_dup_list
import matplotlib


class AxInterface(object):
    pass


class Ax(object):
    SPECS = (
        'n_plot',
        'ax_title',
        'ax_label',
        'ax_label_coords',
        'ax_left_label',
        'ax_left_label_coords',
        'ax_left_label_kwargs',
        'ax_right_label',
        'ax_right_label_coords',
        'ax_right_label_kwargs',
        'map_proj',
        'map_corners',
        'map_res',
        'shiftgrid_start',
        'shiftgrid_cyclic',
        'do_legend',
        'legend_labels',
        'legend_kwargs',
        'x_dim',
        'x_lim',
        'x_ticks',
        'x_ticklabels',
        'x_label',
        'y_dim',
        'y_lim',
        'y_ticks',
        'y_ticklabels',
        'y_label',
        'lat_lim',
        'lat_ticks',
        'lat_ticklabels',
        'lat_label',
        'lon_lim',
        'lon_ticks',
        'lon_ticklabels',
        'lon_label',
        'p_lim',
        'p_ticks',
        'p_ticklabels',
        'p_label',
        'sigma_lim',
        'sigma_ticks',
        'sigma_ticklabels',
        'sigma_label',
        'time_lim',
        'time_ticks',
        'time_ticklabels',
        'time_label',
        'do_mark_x0',
        'do_mark_y0'
    )

    # Which labels to include based on position in the figure.
    labels = {
        'left': {'x_ticklabels': ' ', 'y_ticklabels': True,
                 'x_label': False, 'y_label': True, 'do_colorbar': False},
        'interior': {'x_ticklabels': ' ', 'y_ticklabels': ' ',
                     'x_label': False, 'y_label': False, 'do_colorbar': False},
        'bottomleft': {'x_ticklabels': True, 'y_ticklabels': True,
                       'x_label': True, 'y_label': True, 'do_colorbar': True},
        'bottom': {'x_ticklabels': True, 'y_ticklabels': ' ',
                   'x_label': True, 'y_label': False, 'do_colorbar': True}
    }

    def __init__(self, Fig, ax_num, ax_loc):
        self.fig = Fig
        self.ax_num = ax_num
        self.ax_loc = ax_loc
        self.n_plot = Fig.n_plot[ax_num]
        self.n_data = Fig.n_data[ax_num]
        self.Plot = []

        self._copy_attrs_from_fig()
        # self._set_ax_loc_specs()
        self._set_xy_attrs_to_coords()

    def _traverse_child_tree(self, value, level='data'):
        """Traverse the "tree" of child Plot, and Data objects."""
        assert level in ('ax', 'plot', 'data')
        if level in ('plot', 'data'):
            value = to_dup_list(value, self.n_plot)
            if level in ('data'):
                for i, vi in enumerate(value):
                    value[i] = to_dup_list(vi, self.n_data[i])
        return value

    def _copy_attrs_from_fig(self):
        """Copy the attrs of the parent Fig that correspond to this Ax."""
        for attr in self.SPECS:
            value = getattr(self.fig, attr)[self.ax_num]
            setattr(self, attr, self._traverse_child_tree(value, 'ax'))

        for attr in plot_specs:
            value = getattr(self.fig, attr)[self.ax_num]
            setattr(self, attr, self._traverse_child_tree(value, 'plot'))

        for attr in data_specs:
            value = getattr(self.fig, attr)[self.ax_num]
            setattr(self, attr, self._traverse_child_tree(value, 'data'))

    def _set_ax_loc_specs(self):
        """Set attrs that depend on Ax location within the Fig."""
        # Take the Fig's attr value if it's neeeded; otherwise set False.
        for key, val in self.labels[self.ax_loc].items():
            if val:
                if val == ' ':
                    new_val = ' '
                else:
                    new_val = getattr(self.fig, key, False)

            else:
                new_val = False
            setattr(self, key, new_val)

    def _set_xy_attrs_to_coords(self):
        """
        Set the x and y axis dimensions and related attributes to the values
        specified by the 'x_dim' and 'y_dim' attributes.  E.g. if self.x_dim =
        'lat', then set 'x_lim', etc. equal to 'lat_lim', etc.
        """
        for l, dim in zip(('x', 'y'), ('x_dim', 'y_dim')):
            prefix = getattr(self, dim)
            # prefix being False implies to use the actual x_lim, x_ticks, etc.
            if prefix is False:
                prefix = l
            for attr in ('lim', 'ticks', 'ticklabels', 'label'):
                setattr(self, '_'.join([l, attr]),
                        getattr(self, '_'.join([prefix, attr])))

    def _set_axes_props(self):
        """Set the properties of the matplotlib Axes instance."""
        if self.x_lim:
            if self.x_lim == 'ann_cycle':
                self.x_lim = (1, 12)
                self.x_ticks = range(1, 13)
                self.x_ticklabels = tuple('JFMAMJJASOND')
            self.ax.set_xlim(self.x_lim)
        if self.do_mark_y0:
            self.ax.axhline(color='0.5')
        if self.x_ticks is not False:
            self.ax.set_xticks(self.x_ticks)
        if self.x_ticklabels is not False:
            self.ax.set_xticklabels(self.x_ticklabels, fontsize='x-small')
        if self.x_label:
            self.ax.set_xlabel(self.x_label,
                               fontsize='x-small', labelpad=1)
        if self.y_lim:
            self.ax.set_ylim(self.y_lim)
        if self.do_mark_x0:
            self.ax.axvline(color='0.5')
        if self.y_ticks:
            self.ax.set_yticks(self.y_ticks)
        if self.y_ticklabels:
            self.ax.set_yticklabels(self.y_ticklabels, fontsize='x-small')
        if self.y_label:
            self.ax.set_ylabel(self.y_label, fontsize='x-small', labelpad=-2)

        self.ax.tick_params(labelsize='x-small')
        if not (self.x_dim == 'lon' and self.y_dim == 'lat'):
            self.ax.spines['right'].set_visible(False)
            self.ax.spines['top'].set_visible(False)
            self.ax.xaxis.set_ticks_position('bottom')
            self.ax.yaxis.set_ticks_position('left')

    def _set_axes_labels(self):
        """Create the axis title and other text labels."""
        # Axis title.
        if self.ax_title:
            self.ax.set_title(self.ax_title, fontsize='small')
        # Axis panel labels, i.e. (a), (b), (c), etc.
        if self.ax_label:
            if self.ax_label == 'auto':
                text = '({})'.format(string.ascii_letters[self.ax_num])
            else:
                text = self.ax_label
            self.panel_label = self.ax.text(
                self.ax_label_coords[0], self.ax_label_coords[1],
                text, fontsize='small', transform=self.ax.transAxes
            )
        # Labels to left and/or right of Axis.
        if self.ax_left_label:
            # if self.ax_left_label_rot == 'horizontal':
                # horiz_frac = -0.17
            # else:
            if not self.ax_left_label_kwargs:
                self.ax_left_label_kwargs = dict()
            if not self.ax_left_label_coords:
                default_left_label_coords = (-0.1, 0.5)
                logging.debug("Using default ax_left_label_coords: "
                              "{}".format(default_left_label_coords))
                self.ax_left_label_coords = default_left_label_coords
            self.ax.text(
                self.ax_left_label_coords[0], self.ax_left_label_coords[1],
                self.ax_left_label, transform=self.ax.transAxes,
                **self.ax_left_label_kwargs
            )
        if self.ax_right_label:
            if not self.ax_right_label_kwargs:
                self.ax_right_label_kwargs = dict()
            if not self.ax_right_label_coords:
                default_right_label_coords = (1.1, 0.5)
                logging.debug("Using default ax_right_label_coords: "
                              "{}".format(default_right_label_coords))
                self.ax_right_label_coords = default_right_label_coords
            self.ax.text(
                self.ax_right_label_coords[0], self.ax_right_label_coords[1],
                self.ax_right_label, transform=self.ax.transAxes,
                **self.ax_right_label_kwargs
            )

    def _make_plot_objs(self):
        """Create the Plot object for each plotted element."""
        self.Plot = []
        for n, plot_type in zip(range(self.n_plot), self.plot_type):
            plot_interface = PlotInterface(ax=self, plot_num=n,
                                           plot_type=plot_type)
            if plot_type == 'scatter':
                try:
                    self.Plot.append(Scatter(plot_interface))
                except KeyError:
                    self.Plot.append(None)
            elif plot_type == 'contour':
                self.Plot.append(Contour(plot_interface))
            elif plot_type == 'contourf':
                self.Plot.append(Contourf(plot_interface))
            elif plot_type == 'quiver':
                self.Plot.append(Quiver(plot_interface))
            elif plot_type == 'line':
                self.Plot.append(Line(plot_interface))
            else:
                raise TypeError("Plot type '%s' not recognized."
                                % plot_type)

    def make_plots(self):
        """Call the matplotlib plotting command for each Plot."""
        self._make_plot_objs()
        self._set_axes_props()
        self._set_axes_labels()
        self._handles = []

        # Get the handles for use in the legend.
        # Facilitates excluding extra elements (e.g. x=0 line) from legend.
        for n in range(self.n_plot):
            try:
                handle = self.Plot[n].plot()
            except AttributeError as e:
                logging.warning("\nData not found; skipping plot #{0}.\n"
                                "{1}".format(n, e))
            else:
                # Depending on matplotlib type, have to unpack or not.
                index_cond = isinstance(handle,
                                        (matplotlib.collections.PathCollection,
                                         matplotlib.contour.QuadContourSet,
                                         matplotlib.quiver.Quiver))
                if index_cond:
                    self._handles.append(handle)
                else:
                    self._handles.append(handle[0])

        if self.do_legend:
            if not self.legend_kwargs:
                self.legend_kwargs = dict()
            self.ax.legend(self._handles, self.legend_labels,
                           **self.legend_kwargs)
