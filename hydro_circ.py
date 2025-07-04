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
dt = 10
domain = Domain(mesh, dt, family="RTCF", degree=1)

# we need the latitude and longitude coordinates later
xyz = SpatialCoordinate(mesh)
lon, lat, _ = lonlatr_from_xyz(*xyz)

# set up IO
output = OutputParameters(dirname="hydro_circ")
io = IO(domain, output)

parameters = HydroCircParameters(mesh=mesh)

# Coriolis
fexpr = 2*parameters.Omega*sin(lat)
# moisture
tracers = [WaterVapour(space='DG')]
# equations
eqns = LinearShallowWaterEquations(domain, parameters, fexpr=fexpr,
                                   active_tracers=tracers)

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

# physics_schemes
LinearFriction(eqns)
VerticalVelocity(eqns)
Evaporation(eqns, qs)
Precipitation(eqns)
MoistureDescent(eqns)

for t in eqns.residual:
    print(t.form)

stepper = Timestepper(eqns, ForwardEuler(domain), io)

# =======================================================================
# Our initial conditions are that the height perturbation is zero
# (i.e. the depth of the atmosphere is H) and the horizontal velocity
# is zero. The initial water vapour is 0.7 times the saturation
# function. Where this is above `qC` there will be precipitation, and
# hence vertical velocity, so we compute `P` and then `w`.

# initial water vapour, q, is 0.7 * saturation value
q0 = stepper.fields("water_vapour")
q0.interpolate(0.7 * qs)
# compute P from initial q
# P = Function(q0.function_space()).interpolate(P_expr)
# print_minmax(P)
# compute initial w
# w.interpolate(w_expr)

# =======================================================================
# Now we can timestep!
tmax = 100 * dt
stepper.run(0, tmax)
