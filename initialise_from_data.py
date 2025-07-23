from firedrake import (FiniteElement, TensorProductElement, FunctionSpace,
                       VectorFunctionSpace, Function, op2, Mesh, as_vector,
                       UnitSquareMesh, SpatialCoordinate, VertexOnlyMesh,
                       interpolate, Interpolator, IcosahedralSphereMesh,
                       TestFunction, Cofunction, functionspaceimpl, pi,
                       assemble)
from firedrake.__future__ import interpolate

import numpy as np
from netCDF4 import Dataset


def get_2d_flat_latlon_mesh(mesh):
    """
    Construct a 2D planar latitude-longitude mesh from a spherical mesh.

    Args:
        mesh (:class:`Mesh`): the spherical mesh
    """
    coords_orig = mesh.coordinates
    coords_fs = coords_orig.function_space()

    if coords_fs.extruded:
        cell = mesh._base_mesh.ufl_cell().cellname()
        DG1_hori_elt = FiniteElement("DG", cell, 1, variant="equispaced")
        DG1_vert_elt = FiniteElement("DG", interval, 1, variant="equispaced")
        DG1_elt = TensorProductElement(DG1_hori_elt, DG1_vert_elt)
    else:
        cell = mesh.ufl_cell().cellname()
        DG1_elt = FiniteElement("DG", cell, 1, variant="equispaced")
    vec_DG1 = VectorFunctionSpace(mesh, DG1_elt)
    coords_dg = Function(vec_DG1).interpolate(coords_orig)
    vec_DG1_2d = VectorFunctionSpace(mesh, DG1_elt, dim=2)
    coords_latlon = Function(vec_DG1_2d)
    shapes = {"nDOFs": vec_DG1.finat_element.space_dimension(), 'dim': 2}

    radius = np.min(np.sqrt(coords_dg.dat.data[:, 0]**2 + coords_dg.dat.data[:, 1]**2 + coords_dg.dat.data[:, 2]**2))
    # lat-lon 'x' = atan2(y, x)
    coords_latlon.dat.data[:, 0] = np.arctan2(coords_dg.dat.data[:, 1], coords_dg.dat.data[:, 0])
    # lat-lon 'y' = asin(z/sqrt(x^2 + y^2 + z^2))
    coords_latlon.dat.data[:, 1] = np.arcsin(coords_dg.dat.data[:, 2]/np.sqrt(coords_dg.dat.data[:, 0]**2 + coords_dg.dat.data[:, 1]**2 + coords_dg.dat.data[:, 2]**2))

    # We need to ensure that all points in a cell are on the same side
    # of the branch cut in longitude coords
    
    # This kernel amends the longitude coords so that all longitudes
    # in one cell are close together
    kernel = op2.Kernel("""
#define PI 3.141592653589793
#define TWO_PI 6.283185307179586
void splat_coords(double *coords) {{
    double max_diff = 0.0;
    double diff = 0.0;

    for (int i=0; i<{nDOFs}; i++) {{
        for (int j=0; j<{nDOFs}; j++) {{
            diff = coords[i*{dim}] - coords[j*{dim}];
            if (fabs(diff) > max_diff) {{
                max_diff = diff;
            }}
        }}
    }}

    if (max_diff > PI) {{
        for (int i=0; i<{nDOFs}; i++) {{
            if (coords[i*{dim}] < 0) {{
                coords[i*{dim}] += TWO_PI;
            }}
        }}
    }}
}}
""".format(**shapes), "splat_coords")

    op2.par_loop(kernel, coords_latlon.cell_set,
                 coords_latlon.dat(op2.RW, coords_latlon.cell_node_map()))
    return Mesh(coords_latlon)


def initialise_from_netcdf(dest_mesh, filename):

    data = Dataset(filename)
    Ts_ll = data['temp']
    longitudes = data['longitude']
    latitudes = data['latitude']

    # shift data so that hotspot is in centre of mesh
    T1 = Ts_ll[:, 0:72]
    T2 = Ts_ll[:, 72:]
    Ts_ll = np.hstack((T2, T1))
    lon1 = longitudes[0:72]
    lon2 = longitudes[72:]
    longitudes = np.hstack((lon2, lon1))

    dims = (len(latitudes), len(longitudes))
    latitudes = np.array([-90.0 + i * 180/(len(latitudes)-1) for i in range(len(latitudes))])
    dlon = longitudes[1]-longitudes[0]
    longitudes = np.array([0.0 + i * (360-dlon)/(len(longitudes)-1) for i in range(len(longitudes))])

    # Source mesh is a lon-lat rectangle mesh. Create this using a
    # unit square mesh with quadrilateral cells and adjust coordinates
    # Note: mesh has double the x length so that we don't have
    # problems when chopping the mesh
    src_mesh = UnitSquareMesh(2*dims[1], dims[0], quadrilateral=True)
    Vc = src_mesh.coordinates.function_space()
    x, y = SpatialCoordinate(src_mesh)
    X = Function(Vc).interpolate(as_vector([2*x*360, 180*y-90]))
    src_mesh.coordinates.assign(X)

    # Loop over field values to create a list and store the lon-lat
    # coordinates of the values in the same order
    field_list = []
    coordinates = []
    for ilat in range(dims[0]):
        for ilon in range(dims[1]):
            field_list.append(Ts_ll[ilat, ilon])
            coordinates.append((longitudes[ilon], latitudes[ilat]))
        # second copy of data to extend in x direction to avoid
        # problems with chopping the mesh (i.e. vertex not found)
        for ilon in range(dims[1]):
            field_list.append(Ts_ll[ilat, ilon])
            coordinates.append((longitudes[ilon]+360, latitudes[ilat]))
        # These lines deal with periodic edge by copying data from lon=0
        field_list.append(Ts_ll[ilat, 0])
        coordinates.append((720.0, latitudes[ilat]))

    # Create a vertex only mesh using these coordinates and use the
    # input ordering property to copy in the field values
    vom = VertexOnlyMesh(src_mesh, coordinates)
    P0DG_io = FunctionSpace(vom.input_ordering, "DG", 0)
    field_vomio = Function(P0DG_io)
    field_vomio.dat.data_wo[:] = field_list

    # We now interpolate onto the vertex only mesh that does not have
    # the input ordering
    P0DG = FunctionSpace(vom, "DG", 0)
    field_vom = assemble(interpolate(field_vomio, P0DG, allow_missing_dofs=True))

    # This gets us a CG1 representation of the data on the lon-lat mesh
    Vsrc = FunctionSpace(src_mesh, "CG", 1)
    I = Interpolator(TestFunction(Vsrc), P0DG)
    f_star = field_vom.riesz_representation(riesz_map="l2")
    f_data_star = Cofunction(Vsrc.dual())
    I.interpolate(f_star, adjoint=True, output=f_data_star)
    field = f_data_star.riesz_representation(riesz_map="l2")

    # We now have the input data as a CG1 function on a lon-lat
    # rectangle mesh. We need the data on a spherical mesh.

    # The first step is to adjust the coordinates of the lon-lat mesh
    # to be in radians
    X = Function(Vc).interpolate(as_vector([2*pi*(x-180)/360, 2*pi*y/360]))
    src_mesh.coordinates.assign(X)

    # Get 2D flat lon-lat mesh corresponding to our destination mesh
    dest_mesh_ll = get_2d_flat_latlon_mesh(dest_mesh)

    # Create CG1 function space on lon-lat mesh
    Vdest_ll = FunctionSpace(dest_mesh_ll, "CG", 1)

    # Create CG1 function on lon-lat mesh and interpolate from data mesh
    f_sphere_ll = Function(Vdest_ll).interpolate(field, allow_missing_dofs=True)

    # Create CG1 function on destination spherical mesh that shares
    # the values of that on the lon-lat mesh
    f_sphere = Function(
        functionspaceimpl.WithGeometry.create(
            f_sphere_ll.function_space(), dest_mesh),
        val=f_sphere_ll.topological)

    return f_sphere
