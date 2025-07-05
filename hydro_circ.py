# Import specific functions from Firedrake
from firedrake import (SpatialCoordinate, TestFunction, TrialFunction, norm,
                       Function, dx, lhs, rhs, inner, LinearVariationalProblem,
                       LinearVariationalSolver, Constant, acos, exp,
                       jump, dot, FacetNormal, avg, grad)
from gusto import *
from gusto_physics import *


# Set up mesh, timestep and use Gusto to set up the finite element
# function spaces.
R = 7160000.    # radius of planet (m)
ncells = 16     # number of cells along the edge of each cube face
mesh = GeneralCubedSphereMesh(radius=R, num_cells_per_edge_of_panel=ncells,
                              degree=2)
dt = 100
domain = Domain(mesh, dt, family="RTCF", degree=1)

# we need the latitude and longitude coordinates later
xyz = SpatialCoordinate(mesh)
lon, lat, _ = lonlatr_from_xyz(*xyz)

# set up IO
output = OutputParameters(dirname="hydro_circ",
                          dumpfreq=5,
                          dumplist_latlon=[
                              'D', 'D_error',
                              'u_zonal', 'u_meridional',
                              'water_vapour', 'u_divergence'
                          ])
diagnostic_fields=[ZonalComponent('u'),
                   MeridionalComponent('u'),
                   Divergence('u'),
                   SteadyStateError('D')]
io = IO(domain, output, diagnostic_fields=diagnostic_fields)

# set up the physical parameters (e.g. those use to compute E, P, qA,
# w etc.) Unless specified here, they will take the default values
parameters = HydroCircParameters(mesh=mesh)

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
# fake surface temperature field: a constant Tmin plus Gaussian
# perturbation centered on (lon_c, lat_c)
lon_c = 0
lat_c = 0
Tmin = 230
Tpert = 80

def d(lon1, lat1, lon2, lat2):
    # returns distance on sphere between (lon1, lat1) and (lon2, lat2)
    return acos(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon1-lon2))

Ts = Function(domain.spaces("L2"), name='Ts')
Ts.interpolate(Tmin + Tpert * exp(-d(lon_c, lat_c, lon, lat)**2))

e0 = parameters.e0
L = parameters.L
Rw = parameters.Rw
p0 = parameters.p0
qs = 0.622 * e0 * exp(-L/(Rw*Ts)) / p0

# physics_schemes: this adds the physics terms to the equation
linear_friction = LinearFriction(eqns)
w = VerticalVelocity(eqns)
evap = Evaporation(eqns, qs)
precip = Precipitation(eqns)
qA = MoistureDescent(eqns)

# this makes sure we use the right discretisation for the div(q u) term
transport_methods = [DGUpwind(eqns, 'water_vapour')]

# this makes the transport and physics source terms explicit but does
# the coriolis and gravity terms implicitly - will only have any
# effect if IMEX schemes are used
eqns.residual = eqns.residual.label_map(
    lambda t: t.has_label(transport, source_label),
    map_if_true=lambda t: explicit(t),
    map_if_false=lambda t: implicit(t)
)

stepper = Timestepper(eqns,
                      RK4(domain),
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
tmax = 10000 * dt
stepper.run(0, tmax)
