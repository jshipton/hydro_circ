# Import specific functions from Firedrake
from firedrake import (SpatialCoordinate, TestFunction, TrialFunction, norm,
                       Function, dx, lhs, rhs, inner, LinearVariationalProblem,
                       LinearVariationalSolver, Constant, acos, exp,
                       jump, dot, FacetNormal, avg, grad)
from gusto import *
from gusto_physics import *
from gusto_diagnostics import *
from initialise_from_data import *

explicit_timestepping = True
use_data = False

# Set up mesh, timestep and use Gusto to set up the finite element
# function spaces.
R = 7160000.    # radius of planet (m)
ncells = 16     # number of cells along the edge of each cube face
mesh = GeneralCubedSphereMesh(radius=R, num_cells_per_edge_of_panel=ncells,
                              degree=2)
if explicit_timestepping:
    dt = 100
    dirname = "hydro_circ_explicit"
else:
    dt = 1000
    dirname = "hydro_circ_implicit"
domain = Domain(mesh, dt, family="RTCF", degree=1)

# we need the latitude and longitude coordinates later
xyz = SpatialCoordinate(mesh)
lon, lat, _ = lonlatr_from_xyz(*xyz)

# set up the physical parameters (e.g. those use to compute E, P, qA,
# w etc.) Unless specified here, they will take the default values
parameters = HydroCircParameters(mesh=mesh)
# you can adjust parameters by passing them to HydroCircParameters
# rather than editing the gusto_physics.py file, e.g.:
parameters = HydroCircParameters(mesh=mesh, adjust_qW=True,
                                 adjust_Qcl=True, use_w=True,
                                 conserve_mass=True)

# set up IO
output = OutputParameters(dirname=dirname,
                          dumpfreq=5,
                          dump_vtus=True,
                          dump_nc=True,
                          dumplist_latlon=[
                              'D', 'D_perturbation',
                              'u_zonal', 'u_meridional',
                              'water_vapour', 'u_divergence',
                              'vertical_velocity',
                              'evaporation', 'precipitation'
                          ])
# sneaky hack so that we can set saturation function later...
evap_diag = EvaporationDiagnostic(parameters)
diagnostic_fields=[ZonalComponent('u'),
                   MeridionalComponent('u'),
                   Divergence('u'),
                   Perturbation('D'),
                   evap_diag,
                   PrecipitationDiagnostic(parameters),
                   WDiagnostic(parameters)]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# Coriolis
fexpr = 2*parameters.Omega*sin(lat)
# moisture
tracers = [
    WaterVapour(space='DG', transport_eqn=TransportEquationType.conservative)
]
# equations
eqns = LinearShallowWaterEquations(domain, parameters, fexpr=fexpr,
                                   active_tracers=tracers)
# hack to add in moisture tracer
tracer_names = [tracer.name for tracer in tracers]
for i, (test, field_name) in enumerate(zip(eqns.tests, eqns.field_names)):
    for tracer_name in tracer_names:
        if field_name == tracer_name:
            prog = split(eqns.X)[i]
            eqns.residual += time_derivative(
                subject(prognostic(inner(prog, test)*dx,
                                   field_name), eqns.X)
                )

# Add transport of moisture
eqns.residual += eqns.generate_tracer_transport_terms(tracers)

# compute saturation function
Ts = Function(domain.spaces("L2"), name='Ts')
if use_data:
    Ts.interpolate(initialise_from_netcdf(mesh, "initial_surf_temp_280c.nc"))
else:
    # fake surface temperature field: a constant Tmin plus Gaussian
    # perturbation centered on (lon_c, lat_c)
    lon_c = 0
    lat_c = 0
    Tmin = 230
    Tpert = 80

    def d(lon1, lat1, lon2, lat2):
        # returns distance on sphere between (lon1, lat1) and (lon2, lat2)
        return acos(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon1-lon2))

    Ts.interpolate(Tmin + Tpert * exp(-d(lon_c, lat_c, lon, lat)**2))

e0 = parameters.e0
L = parameters.L
Rw = parameters.Rw
p0 = parameters.p0
qs = 0.622 * e0 * exp(-L/(Rw*Ts)) / p0
# ...setting saturation function like I said I would earlier
evap_diag.qs = qs

# physics_schemes: this adds the physics terms to the equation
linear_friction = LinearFriction(eqns)
precip = Precipitation(eqns)
w = VerticalVelocity(eqns)
evap = Evaporation(eqns, qs)
qA = MoistureDescent(eqns)

# this makes sure we use the right discretisation for the div(q u) term
transport_methods = [DGUpwind(eqns, 'water_vapour')]

if explicit_timestepping:
    # use an RK4 timestepper
    stepper = Timestepper(eqns,
                          RK4(domain),
                          io, spatial_methods=transport_methods)
else:
    # this makes the transport and physics source terms explicit but does
    # the coriolis and gravity terms implicitly - will only have any
    # effect if IMEX schemes are used
    eqns.residual = eqns.residual.label_map(
        lambda t: t.has_label(transport),
        map_if_true=lambda t: explicit(t)
    )
    eqns.residual = eqns.residual.label_map(
        lambda t: any(t.has_label(pressure_gradient, coriolis, return_tuple=True)),
        map_if_true=lambda t: implicit(t)
    )

    # use an IMEX timestepper
    stepper = Timestepper(eqns,
                          IMEX_SSP3(domain),
                          io, spatial_methods=transport_methods)

stepper.set_reference_profiles([('D', parameters.H)])

# =======================================================================
# Our initial conditions are that the height perturbation is zero
# (i.e. the depth of the atmosphere is H) and the horizontal velocity
# is zero. The initial water vapour is 0.7 times the saturation
# function. Where this is above `qC` there will be precipitation, and
# hence vertical velocity, so we compute `P` and then `w`.

# initial water vapour, q, is 0.7 * saturation value
q0 = stepper.fields("water_vapour")
q0.interpolate(0.7 * qs)
D0 = stepper.fields("D")
D0.interpolate(parameters.H)

# =======================================================================
# Now we can timestep!
tmax = 100 * dt
stepper.run(0, tmax)
