from tomplot import *
import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt

from firedrake import (trisurf, FunctionSpace, SpatialCoordinate, Function,
                       pi, dot)
from gusto import rotated_lonlatr_vectors, lonlatr_from_xyz


def plot_field(field, figname):
    # function to plot the field and save or show the plot
    tsurf = trisurf(field)
    plt.colorbar(tsurf)
    plt.title(field.name())
    plt.savefig(figname)


def plot_field_latlon(field, figname):
    # function to plot the field and save or show the plot

    # We need to regrid onto lon-lat grid -- specify that here
    lon_1d = np.linspace(-180.0, 180.0, 120)
    lat_1d = np.linspace(-90, 90, 120)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='ij')

    projection=ccrs.Robinson()
    contour_method = 'contour'

    # set up figure
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    # get function space and field data
    if field.name() == 'u':
      mesh = field.function_space().mesh()
      V = FunctionSpace(mesh, "DG", 1)
      xyz = SpatialCoordinate(mesh)
      rotated_pole = (0.0, pi/2)
      e_lon, e_lat, _ = rotated_lonlatr_vectors(xyz, rotated_pole)
      u_zonal = Function(V).interpolate(dot(field, e_lon)).dat.data_ro
      u_merid = Function(V).interpolate(dot(field, e_lat)).dat.data_ro
      field_data = np.sqrt(u_zonal**2 + u_merid**2)
    else:
      V = field.function_space()
      field_data = field.dat.data_ro

    # get lat lon coordinates of DOFs
    x, y, z = SpatialCoordinate(V.mesh())
    lon, lat, _ = lonlatr_from_xyz(x, y, z)
    coords_X = Function(V).interpolate(180.0 / np.pi * lon).dat.data_ro
    coords_Y = Function(V).interpolate(180.0 / np.pi * lat).dat.data_ro

    # map field data to regular lat-lon grid
    field_data = regrid_horizontal_slice(
        lon_2d, lat_2d, coords_X, coords_Y, field_data, periodic_fix='sphere'
    )

    # generate 10 contours between min and max values of field
    contours = np.linspace(field_data.min(), field_data.max(), 10)

    # plot contours
    cmap, lines = tomplot_cmap(contours)
    cf, _ = plot_contoured_field(
        ax, lon_2d, lat_2d, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines, projection=projection
    )

    # add colourbar and title
    add_colorbar_ax(
        ax, cf, field.name(), location='bottom', cbar_labelpad=-10,
    )
    tomplot_field_title(ax, None, minmax=True, field_data=field_data)

    # add quivers if field is velocity
    if field.name() == 'u':
        # Need to re-grid to lat-lon grid to get sensible looking quivers
        regrid_zonal_data = regrid_horizontal_slice(
            lon_2d, lat_2d, coords_X, coords_Y, u_zonal,
            periodic_fix='sphere'
        )
        regrid_meridional_data = regrid_horizontal_slice(
            lon_2d, lat_2d, coords_X, coords_Y, u_merid,
            periodic_fix='sphere'
        )
        plt.quiver(lon_2d[::5], lat_2d[::5], regrid_zonal_data[::5]/field_data[::5], regrid_meridional_data[::5]/field_data[::5])
        #plot_field_quivers(
        #    ax, lon_2d, lat_2d, regrid_zonal_data, regrid_meridional_data,
        #    magnitude_filter=1.0, scale=10.,
        #    projection=ccrs.PlateCarree()
        #)

    plt.savefig(figname)


def plot_u_components(field, figname):
    # function to plot the field and save or show the plot
    assert field.name() == 'u'

    # We need to regrid onto lon-lat grid -- specify that here
    lon_1d = np.linspace(-180.0, 180.0, 120)
    lat_1d = np.linspace(-90, 90, 120)
    lon_2d, lat_2d = np.meshgrid(lon_1d, lat_1d, indexing='ij')

    projection=ccrs.Robinson()
    contour_method = 'contour'

    # set up figure
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=projection)

    # get function space and field data
    mesh = field.function_space().mesh()
    V = FunctionSpace(mesh, "DG", 1)
    xyz = SpatialCoordinate(mesh)
    rotated_pole = (0.0, pi/2)
    e_lon, e_lat, _ = rotated_lonlatr_vectors(xyz, rotated_pole)
    u_zonal = Function(V).interpolate(dot(field, e_lon)).dat.data_ro
    u_merid = Function(V).interpolate(dot(field, e_lat)).dat.data_ro

    # get lat lon coordinates of DOFs
    x, y, z = SpatialCoordinate(V.mesh())
    lon, lat, _ = lonlatr_from_xyz(x, y, z)
    coords_X = Function(V).interpolate(180.0 / np.pi * lon).dat.data_ro
    coords_Y = Function(V).interpolate(180.0 / np.pi * lat).dat.data_ro

    # map field data to regular lat-lon grid
    field_data = regrid_horizontal_slice(
        lon_2d, lat_2d, coords_X, coords_Y, u_zonal, periodic_fix='sphere'
    )

    # generate 10 contours between min and max values of field
    contours = np.linspace(field_data.min(), field_data.max(), 10)

    # plot contours
    cmap, lines = tomplot_cmap(contours)
    cf, _ = plot_contoured_field(
        ax, lon_2d, lat_2d, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines, projection=projection
    )

    # add colourbar and title
    add_colorbar_ax(
        ax, cf, field.name(), location='bottom', cbar_labelpad=-10,
    )
    tomplot_field_title(ax, None, minmax=True, field_data=field_data)

    plt.savefig(figname+'_zonal')

    # set up figure
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=projection)    

    # map field data to regular lat-lon grid
    field_data = regrid_horizontal_slice(
        lon_2d, lat_2d, coords_X, coords_Y, u_merid, periodic_fix='sphere'
    )

    # generate 10 contours between min and max values of field
    contours = np.linspace(field_data.min(), field_data.max(), 10)

    # plot contours
    cmap, lines = tomplot_cmap(contours)
    cf, _ = plot_contoured_field(
        ax, lon_2d, lat_2d, field_data, contour_method, contours,
        cmap=cmap, line_contours=lines, projection=projection
    )

    # add colourbar and title
    add_colorbar_ax(
        ax, cf, field.name(), location='bottom', cbar_labelpad=-10,
    )
    tomplot_field_title(ax, None, minmax=True, field_data=field_data)

    plt.savefig(figname+'_merid')


def print_minmax(field):
    # function to print the min and max of field in a nice way
    print(f"min and max of {field.name()}: {field.dat.data.min()}, {field.dat.data.max()}")
