"""Classes for creating multi-panel figures using data generated via aospy."""
import logging

import scipy.stats
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import mpl_toolkits.basemap
import xarray as xr

from . import default_colormap, PFULL_STR
from aospy.calc import Calc, CalcInterface
from aospy.io import to_dup_list
from aospy.operator import Operator
from aospy.utils import to_hpa


class PlotInterface(object):
    """Interface to Plot class."""
    SPECS = (
        'proj',
        'model',
        'run',
        'ens_mem',
        'var',
        'level',
        'region',
        'date_range',
        'intvl_in',
        'intvl_out',
        'dtype_in_time',
        'dtype_in_vert',
        'dtype_out_time',
        'dtype_out_vert',
        'do_subtract_mean',
        'mult_factor',
        'mask_unphysical',
        'smooth_data'
    )

    def __init__(self, ax=False, plot_num=1, plot_type=False):
        self.ax = ax
        self.plot_num = plot_num
        self.plot_type = plot_type
        self._copy_attrs_from_ax(ax)

    def _copy_attrs_from_ax(self, ax):
        """Copy the attrs of the parent Ax corresponding to this Plot."""
        self.fig = ax.fig
        try:
            self.n_data = ax.n_data[self.plot_num]
        except AttributeError:
            self.n_data = 1

        for attr in self.SPECS:
            value = getattr(ax, attr)[self.plot_num]
            tree_value = self._traverse_child_tree(value, 'plot')
            setattr(self, attr, tree_value)

        for attr in data_specs:
            value = getattr(ax, attr)[self.plot_num]
            tree_value = self._traverse_child_tree(value, 'data')
            setattr(self, attr, tree_value)

    def _traverse_child_tree(self, value, level='data'):
        """Traverse the "tree" of child Data objects."""
        assert level in ('plot', 'data')
        if level == 'data':
            value = to_dup_list(value, self.n_data)
        return value


class Plot(object):
    def __init__(self, plot_interface):
        """Class for plotting single data element."""
        self.__dict__ = vars(plot_interface)

        if isinstance(self.var[0], tuple):
            self.var = self.var[0]
            self.n_data = len(self.var)

        self.calc = self._make_calc_obj()
        # Pass the calcs to the _load_data method.
        self.data = [self._load_data(calc, n)
                     for n, calc in enumerate(self.calc)]
        # Strip extra dimensions as necessary.
        if len(self.data) == 1:
            self.data = self.data[0]
        else:
            self.data = [d.squeeze() for d in self.data]
        # _set_coord_arrays() below needs Calc, not Operator, objects.
        for i, calc in enumerate(self.calc):
            if isinstance(calc, Operator):
                self.calc[i] = calc.objects[0]

        # self._apply_data_transforms()

        self._set_coord_arrays()

        if self.ax.x_dim == 'lon' and self.ax.y_dim == 'lat':
            self.basemap = self._make_basemap()
            self.backend = self.basemap
        else:
            self.backend = self.ax.ax

    def plot(self):
        return NotImplementedError("plot() is an abstract method: it is only "
                                   "implemented for the classes that "
                                   "inherit from Plot.")

    def _make_calc_obj(self):
        calc_obj = []
        for i in range(self.n_data):
            if isinstance(self.run[i], Operator):
                calcs = [Calc(CalcInterface(
                    proj=self.proj[i], model=self.model[i],
                    run=run,
                    ens_mem=self.ens_mem[i], var=self.var[i],
                    date_range=self.date_range[i], region=self.region[i],
                    intvl_in=self.intvl_in[i], intvl_out=self.intvl_out[i],
                    dtype_in_time=self.dtype_in_time[i],
                    dtype_in_vert=self.dtype_in_vert[i],
                    dtype_out_time=self.dtype_out_time[i],
                    dtype_out_vert=self.dtype_out_vert[i], level=self.level[i],
                    verbose=self.fig.verbose
                )) for run in self.run[i].objects]
                calc_obj.append(Operator(self.run[i].operator, calcs))
            else:
                calc_obj.append(Calc(CalcInterface(
                    proj=self.proj[i], model=self.model[i], run=self.run[i],
                    ens_mem=self.ens_mem[i], var=self.var[i],
                    date_range=self.date_range[i], region=self.region[i],
                    intvl_in=self.intvl_in[i], intvl_out=self.intvl_out[i],
                    dtype_in_time=self.dtype_in_time[i],
                    dtype_in_vert=self.dtype_in_vert[i],
                    dtype_out_time=self.dtype_out_time[i],
                    dtype_out_vert=self.dtype_out_vert[i], level=self.level[i],
                    verbose=self.fig.verbose
                )))
        return calc_obj

    def _set_coord_arrays(self):
        """Set the arrays holding the x- and y-coordinates."""
        array_names = {'lat': 'lat', 'lon': 'lon', 'p': 'level',
                       'sigma': 'pfull', 'time': 'time_bounds',
                       'x': 'x', 'y': 'y'}
        if self.n_data == 1:
            mod_inds = (0, 0)
        else:
            mod_inds = (0, 1)
        for dim, data, lim, i in zip(('x_dim', 'y_dim'), ('x_data', 'y_data'),
                                     ('x_lim', 'y_lim'), mod_inds):
            array_key = getattr(self.ax, dim)

            if isinstance(self.calc[i], Calc):
                calc = self.calc[i]
            elif isinstance(self.calc[i][0], Calc):
                calc = self.calc[i][0]
            else:
                msg = ("Couldn't find the Calc object for the plot "
                       "object `{}`".format(self))
                raise ValueError(msg)

            if array_key in ('x', 'y'):
                if len(self.data) == 1:
                    array = self.data[0]
                else:
                    array = self.data
            elif array_key == 'time':
                # Hack to get timeseries plotted.
                # TODO: clean this up.
                array = np.arange(calc.start_date.year, calc.end_date.year + 1)
                if lim == 'ann_cycle':
                    array = np.arange(1, 13)
            else:
                array = getattr(calc.model[0], array_names[array_key], None)

            if array_key == 'p':
                # Hack to get pressure data if not found previously.
                # TODO: clean this up.
                if array is None:
                    try:
                        array = self.x_data.level
                    except AttributeError:
                        array = self.y_data.level
                array = to_hpa(array)

            setattr(self, data, array)

    def _make_basemap(self):
        if self.ax.x_lim:
            llcrnrlon, urcrnrlon = self.ax.x_lim
        else:
            llcrnrlon, urcrnrlon = -180, 180
        if self.ax.y_lim:
            llcrnrlat, urcrnrlat = self.ax.y_lim
        else:
            llcrnrlat, urcrnrlat = -90, 90

        return mpl_toolkits.basemap.Basemap(
            projection=self.ax.map_proj, resolution=self.ax.map_res,
            ax=self.ax.ax, llcrnrlon=llcrnrlon, llcrnrlat=llcrnrlat,
            urcrnrlon=urcrnrlon, urcrnrlat=urcrnrlat
        )

    @staticmethod
    def regrid_to_avg_coords(dim, *arrs):
        """Average the coordinate arrays of two DataArrays or Dataset."""
        template = arrs[0][dim]
        avg = xr.DataArray(np.zeros(template.shape), dims=template.dims,
                           coords=template.coords)
        for arr in arrs:
            avg += arr[dim]
        avg /= len(arrs)
        for arr in arrs:
            arr[dim] = avg
        return arrs

    @classmethod
    def _perform_oper(cls, arr1, arr2, operator, region=False):
        # Only regrid data on model-native coordinates, not pressure.
        if region and any((getattr(arr, pfs, False) for arr in (arr1, arr2)
                           for pfs in (PFULL_STR, PFULL_STR + '_ref'))):
            try:
                arr1, arr2 = cls.regrid_to_avg_coords(PFULL_STR, arr1, arr2)
            except KeyError:
                arr1, arr2 = cls.regrid_to_avg_coords(PFULL_STR + '_ref',
                                                      arr1, arr2)
        return eval('arr1' + operator + 'arr2')

    def _load_data(self, calc, n):
        if isinstance(calc, Operator):
            region = self.region[n]
            data = tuple(
                [cl.load(self.dtype_out_time[n],
                         dtype_out_vert=self.dtype_out_vert[n],
                         region=self.region[n], time=False, vert=self.level[n],
                         lat=False, lon=False, plot_units=True,
                         mask_unphysical=self.mask_unphysical)
                 for cl in calc.objects]
            )
            ans = self._perform_oper(data[0], data[1], calc.operator,
                                     region=region,)
            # Combine masks of the two inputs.
            try:
                joint_mask = (np.ma.mask_or(data[0].mask, data[1].mask),)
            except AttributeError:
                return ans
            else:
                return (np.ma.array(ans, mask=joint_mask),)

        if isinstance(calc, (list, tuple)):
            return tuple(
                [cl.load(dto, dtype_out_vert=dtv, region=reg, time=False,
                         vert=lev, lat=False, lon=False, plot_units=True,
                         mask_unphysical=self.mask_unphysical)
                 for cl, dto, dtv, reg, lev in zip(
                         calc, self.dtype_out_time, self.dtype_out_vert,
                         self.region, self.level
                 )]
            )
        return calc.load(self.dtype_out_time[n],
                         dtype_out_vert=self.dtype_out_vert[n],
                         region=self.region[n], time=False,
                         vert=self.level[n],
                         lat=False,
                         lon=False,
                         plot_units=True,
                         mask_unphysical=self.mask_unphysical)

    def _subtract_mean(self, data):
        return np.subtract(data, np.mean(data))

    def _apply_data_transforms(self):
        """Apply any specified transformations to the data once loaded."""
        transforms = {'do_subtract_mean': self._subtract_mean}
        for attr, method in transforms.items():
            for data, do_method in zip(['x_data', 'y_data'],
                                       getattr(self, attr)):
                if do_method:
                    setattr(self, data, method(getattr(self, data)))
        return data

    def prep_data_for_basemap(self):
        # if self.ax.shiftgrid_start:
        #     lon0 = 181.25
        # else:
        #     lon0 = self.ax.shiftgrid_start
        lon0 = 181.25
        self.lons = self.x_data
        self.lats = self.y_data

        self.plot_data = []

        if isinstance(self.data, xr.Dataset):
            loop_data = [self.data[self.var[0].name].values]
        else:
            loop_data = self.data
        for data in loop_data:
            d, self.plot_lons = mpl_toolkits.basemap.shiftgrid(
                lon0, data, self.x_data,
                start=self.ax.shiftgrid_start, cyclic=self.ax.shiftgrid_cyclic
            )
            self.plot_data.append(d)
        if len(self.plot_data) == 1:
            self.plot_data = self.plot_data[0]

        self.x_data, self.y_data = self.basemap(*np.meshgrid(self.plot_lons,
                                                             self.y_data))
        if self.do_mask_oceans:
            self.plot_data = mpl_toolkits.basemap.maskoceans(
                self.x_data, self.y_data, self.plot_data,
                inlands=False, resolution='c'
            )

    def plot_rectangle(self, x_min, x_max, y_min, y_max, **kwargs):
        xs = [x_min, x_max, x_max, x_min, x_min]
        ys = [y_min, y_min, y_max, y_max, y_min]
        return self.backend.plot(xs, ys, latlon=True, linewidth=0.8, **kwargs)

    def apply_basemap(self, basemap):
        """Apply basemap extras: coastlines, etc."""
        basemap.drawcoastlines(linewidth=0.1, color='k')
        basemap.drawmapboundary(linewidth=0.1, fill_color='0.85')
        if self.latlon_rect:
            self.plot_rectangle(*self.latlon_rect)

    def apply_colormap(self, colormap):
        if colormap == 'default':
            try:
                colormap = self.var[0].colormap
            except AttributeError:
                colormap = default_colormap
        self.handle.set_cmap(colormap)

    def corr_coeff(self, x_data, y_data, print_corr=True, print_pval=True):
        """Compute the Pearson correlation coefficient and plot it."""
        pearsonr, p_val = scipy.stats.pearsonr(x_data, y_data)
        if print_corr:
            self.ax.ax.text(0.3, 0.95, r'$r=$ %.2f' % pearsonr,
                            transform=self.ax.ax.transAxes, fontsize='x-small')
        if print_pval:
            self.ax.ax.text(0.3, 0.9, r'$p=$ %.3f' % p_val,
                            transform=self.ax.ax.transAxes, fontsize='x-small')
        return pearsonr, p_val

    def best_fit_line(self, print_slope=True, print_y0=True):
        """Plot the best fit line to the data."""
        # Enforce a dtype of float to ensure data plays nice with np.polyfit
        x_data = np.array(self.x_data).astype('float64')
        y_data = np.array(self.y_data).astype('float64')
        best_fit = np.polyfit(x_data, y_data, 1)
        x_lin_fit = [-1e3, 1e3]

        def lin_fit(m, x, b):
            return [m*xx + b for xx in x]
        self.backend.plot(x_lin_fit, lin_fit(best_fit[0], x_lin_fit,
                                             best_fit[1]), 'k')
        if print_slope:
            self.ax.ax.text(0.3, 0.1, r'slope = %0.2f' % best_fit[0],
                            transform=self.ax.ax.transAxes, fontsize='x-small')
        if print_y0:
            self.ax.ax.text(0.3, 0.05, r'y0 = %0.2f' % best_fit[1],
                            transform=self.ax.ax.transAxes, fontsize='x-small')
        return best_fit

    def _get_color_from_cmap(self, value):
        """Get a color along some specified fraction of a colormap."""
        if isinstance(self.colormap, str):
            try:
                self.colormap = getattr(cm, self.colormap)
            except AttributeError:
                logging.warning("Desired colormap '{0}' not found.  Using "
                                "'RdBu_r' by default.".format(self.colormap))
                self.colormap = cm.RdBu_r
        return self.colormap(value)


class Contour(Plot):
    def __init__(self, plot_interface):
        """Contour plot."""
        Plot.__init__(self, plot_interface)
        self.plot_func = self.backend.contour
        self.plot_func_kwargs = self.contour_kwargs
        self.cntr_lvls = np.linspace(self.min_cntr, self.max_cntr,
                                     self.num_cntr + 1)

    def _prep_data(self):
        if self.basemap:
            self.prep_data_for_basemap()
        else:
            self.plot_data = self.data[0]
        if self.mult_factor[0]:
            self.plot_data = np.multiply(float(self.mult_factor[0]),
                                         self.plot_data)

    def plot(self):
        self._prep_data()
        self.handle = self.plot_func(self.x_data, self.y_data, self.plot_data,
                                     self.cntr_lvls, **self.plot_func_kwargs)
        if self.contour_labels:
            plt.gca().clabel(self.handle, fontsize=7, fmt='%1d')
        if self.colormap:
            self.apply_colormap(self.colormap)
        if self.do_colorbar:
            self.fig._make_colorbar(self.ax)
        if self.basemap:
            self.apply_basemap(self.basemap)

        return self.handle


class Contourf(Contour):
    """Filled contour ('contourf') plot."""
    def __init__(self, plot_interface):
        Contour.__init__(self, plot_interface)
        self.plot_func = self.backend.contourf
        self.plot_func_kwargs = self.contourf_kwargs


class Line(Plot):
    """Line plot."""
    def __init__(self, plot_interface):
        Plot.__init__(self, plot_interface)

    def plot(self):
        """Plot the line plot."""
        x_data = self.x_data.copy()
        y_data = self.y_data.copy()
        # TODO: refactor to avoid repetitive code.
        if self.mult_factor[0]:
            if self.ax.x_dim == 'x':
                x_data = np.multiply(float(self.mult_factor[0]), x_data)
            if self.ax.y_dim == 'y':
                y_data = np.multiply(float(self.mult_factor[0]), y_data)
        # TODO: generalize this beyond time-data (c.f. 'year=...')
        if self.smooth_data[0]:
            if self.ax.x_dim == 'x':
                x_data = x_data.rolling(center=True,
                                        year=self.smooth_data[0]).mean()
            if self.ax.y_dim == 'y':
                y_data = y_data.rolling(center=True,
                                        year=self.smooth_data[0]).mean()

        if not self.plot_kwargs:
            self.plot_kwargs = {}
        plot_kwargs = self.plot_kwargs.copy()
        if self.colormap:
            rgba = self._get_color_from_cmap(plot_kwargs['color'])
        else:
            rgba = plot_kwargs.get('color', 'blue')
        plot_kwargs.pop('color', None)
        self.handle = self.backend.plot(x_data, y_data, color=rgba,
                                        **plot_kwargs)
        if self.do_colorbar:
            self.fig._make_colorbar(self.ax)

        return self.handle


class Scatter(Plot):
    def __init__(self, plot_interface):
        Plot.__init__(self, plot_interface)
        self.x_data = np.squeeze(self.data[0])
        self.y_data = np.squeeze(self.data[1])
        self._apply_data_transforms()

    def plot(self):
        scatter_kwargs = self.scatter_kwargs.copy()
        if self.colormap == 'norm':
            rgba = self._get_color_from_cmap(scatter_kwargs['c'])
        else:
            rgba = scatter_kwargs['c']
        scatter_kwargs.pop('c', None)
        self.handle = self.backend.scatter(self.x_data, self.y_data, c=rgba,
                                           **scatter_kwargs)
        if self.do_best_fit_line:
            self.best_fit_line(print_slope=self.print_best_fit_slope)

        if self.print_corr_coeff:
            self.corr_coeff(self.x_data, self.y_data)

        if self.do_colorbar:
            self.fig._make_colorbar(self.ax)

        return self.handle


class Quiver(Plot):
    """Quiver (i.e. vector) plot."""
    def __init__(self, plot_interface):
        Plot.__init__(self, plot_interface)

    def prep_quiver(self):
        if not self.quiver_n_lon:
            self.quiver_n_lon = self.plot_lons.size
        if not self.quiver_n_lat:
            self.quiver_n_lat = self.lats.size
        return self.backend.transform_vector(
            self.plot_data[0], self.plot_data[1], self.plot_lons,
            self.lats.values, self.quiver_n_lon, self.quiver_n_lat,
            returnxy=True
        )

    def plot(self):
        if self.basemap:
            self.prep_data_for_basemap()
        else:
            self.plot_data = self.data[0]

        u, v, x, y = self.prep_quiver()
        self.handle = self.backend.quiver(x, y, u, v, **self.quiver_kwargs)

        if self.do_quiverkey:
            self.quiverkey = plt.quiverkey(self.handle, *self.quiverkey_args,
                                           **self.quiverkey_kwargs)
        if self.basemap:
            self.apply_basemap(self.basemap)

        if self.do_colorbar:
            self.fig._make_colorbar(self.ax)

        return self.handle
