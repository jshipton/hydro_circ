

# Import specific functions from Firedrake
from firedrake import (SpatialCoordinate, TestFunction, TrialFunction, norm,
                       Function, dx, lhs, rhs, inner, LinearVariationalProblem,
                       LinearVariationalSolver, Constant, acos, exp,
                       jump, dot, FacetNormal, avg, grad)
from gusto import *
from plotting import *


# =======================================================================
# Planetary parameters relevant to Proxima B.
R = 7160000.        # radius of planet (m)
# Omega = 6.501e-6    # rotation rate of planet (rad s^-1)
Omega = 0.
g = 10.9            # gravitational acceleration (m s^-2)
H = 2500.           # depth of atmospheric boundary layer (mean fluid depth) (m)

# =======================================================================
# Coefficients used in equations
r = 3.6e-5          # linear coefficient of friction (N kg^-1 m^-1 s)

# =======================================================================
# Large scale moisture parameters
q_ut = 0.004        # upper-tropospheric specific humidity (kg kg^-1)
qW = q_ut           # tropically averaged descending moisture (kg
                    # kg^-1). q_ut for now, but should be somewhat
                    # higher probably.

# =======================================================================
# Parameters used to compute evaporation and precipitation
rho0 = 1                     # constant density of boundary layer air (kg m^-3)
cH = 1e-3                    # bulk coefficient (no units)
L = 2.5e6                    # latent heat of condensation (J kg^-1)
Rw = 416                     # specific gas constant for water vapour
                             # (J K^-1 kg^-1)
e0 = 2300*exp(L/(Rw*293.))   # reference saturation vapour pressure (Pa)
p0 = 1e5                     # reference pressure (Pa)
mB = 0.006029                # boundary layer overturning timescale
                             # (kg m^-2 s^-1)

# =======================================================================
# Parameters used to compute vertical velocity w
Cp = 1005.                   # specific heat capacity at constant
                             # pressure of dry air (J K^-1 kg^-1)
dtheta = 50                  # change in potential temperature across
                             # troposphere (K)
Qcl = 100.                   # net combined radiative-sensible cooling
                             # of free atmosphere (W m^-2)
qC = 0.018                   # critical specific humidity for
                             # initiation of convection (kg kg^-1)

# =======================================================================
# Set up mesh, timestep and use Gusto to set up the finite element
# function spaces.
ncells = 16     # number of cells along the edge of each cube face
mesh = GeneralCubedSphereMesh(radius=R, num_cells_per_edge_of_panel=ncells,
                              degree=2)
dt = 10
domain = Domain(mesh, dt, family="RTCF", degree=1)

# extract function spaces from domain
Vu = domain.spaces("HDiv")
Vdg = domain.spaces("L2")
Vcg = domain.spaces("H1")

# we need the latitude and longitude coordinates later
xyz = SpatialCoordinate(mesh)
lon, lat, _ = lonlatr_from_xyz(*xyz)

# =======================================================================
# create the functions we need and name them
w = Function(Vdg, name="w")       # vertical velocity
h = Function(Vdg, name='h')       # depth perturbation
u = Function(Vu, name='u')        # velocity
q = Function(Vdg, name='q')       # moisture
qA = Function(Vdg, name='qA')     #
E = Function(Vdg, name='E')       # evaporation
P = Function(Vdg, name='P')       # precipitation


# Now we setup functions for the Coriolis parameter, surface
# temperature and saturation function.

# Coriolis
fexpr = 2*Omega*sin(lat)
f = Function(Vcg).interpolate(fexpr)

# fake surface temperature field: a constant Tmin plus Gaussian
# perturbation centered on (lon_c, lat_c)
lon_c = 0
lat_c = 0
Tmin = 230
Tpert = 80

def d(lon1, lat1, lon2, lat2):
    # returns distance on sphere between (lon1, lat1) and (lon2, lat2)
    return acos(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon1-lon2))

Ts = Function(Vdg, name='Ts')
Ts.interpolate(Tmin + Tpert * exp(-d(lon_c, lat_c, lon, lat)**2))

# saturation function
qs = 0.622 * e0 * exp(-L/(Rw*Ts)) / p0
plot_field(Ts, "Ts")

# =======================================================================
# Now set up the finite element form of the equations we are going to solve.
# =======================================================================
# setup the h equation and solver
test_h = TestFunction(Vdg)
trial_h = TrialFunction(Vdg)
h0 = Function(Vdg)
n = FacetNormal(mesh)
dx_max = 2*pi*R/(4*ncells)
mu = 5/dx_max
h_eqn = test_h * (trial_h - h0) * dx + dt * (
    - test_h * w * dx
    + (g*H/r) * (
        inner(grad(test_h), grad(h0)) * dx
        - inner(2*avg(h0 * n), avg(grad(test_h))) * dS
        - inner(avg(grad(h0)), 2*avg(test_h * n)) * dS
        + mu * inner(2*avg(h0 * n), 2*avg(test_h * n)) * dS
    )
    + test_h * (H/r) * div(f * domain.perp(u)) * dx
)
h_lhs = lhs(h_eqn)
h_rhs = rhs(h_eqn)
h_prob = LinearVariationalProblem(h_lhs, h_rhs, h)
h_solver = LinearVariationalSolver(h_prob)


# =======================================================================
# setup the u equation and solver
test_u = TestFunction(Vu)
trial_u = TrialFunction(Vu)
u0 = Function(Vu)
u_eqn = inner(test_u, (trial_u - u0)) * dx + dt * (
    inner(test_u, f*domain.perp(u0) + r*u0) * dx + g * div(test_u) * h * dx
)
u_lhs = lhs(u_eqn)
u_rhs = rhs(u_eqn)
u_prob = LinearVariationalProblem(u_lhs, u_rhs, u)
u_solver = LinearVariationalSolver(u_prob)


# =======================================================================
# setup the q equation and solver
test_q = TestFunction(Vdg)
trial_q = TrialFunction(Vdg)
q0 = Function(Vdg)
un = 0.5*(dot(u, n) + abs(dot(u, n)))
q_eqn = test_q * (trial_q - q0) * dx + dt * (
    - div(test_q * u) * q0 * dx
    + dot(jump(test_q), (un('+')*q0('+') - un('-')*q0('-'))) * dS
    + test_q * ((q0 - qA) * div(u)
    - (E - P)/(rho0*H)) * dx
    )
q_lhs = lhs(q_eqn)
q_rhs = rhs(q_eqn)
q_prob = LinearVariationalProblem(q_lhs, q_rhs, q)
q_solver = LinearVariationalSolver(q_prob)

# =======================================================================
# Now we define some expressions for calculating various fields -
# hopefully the naming is obvious: e.g. `E_expr` is the expression
# used to calculate the evaporation `E`."""
w_expr = (L * P - Qcl) / (rho0 * Cp * dtheta)
qA_expr = conditional(w < 0, qW, 0)
E_expr = conditional(qs > q0, rho0 * cH * sqrt(dot(u, u)) * (qs - q), 0)
P_expr = conditional(q0 > qC, mB * (q0 - q_ut), 0)

# =======================================================================
# Our initial conditions are that the height perturbation is zero
# (i.e. the depth of the atmosphere is H) and the horizontal velocity
# is zero. The initial water vapour is 0.7 times the saturation
# function. Where this is above `qC` there will be precipitation, and
# hence vertical velocity, so we compute `P` and then `w`.

# initial water vapour, q, is 0.7 * saturation value
q0.interpolate(0.7 * qs)
q_init = Function(q0.function_space()).assign(q0)
print_minmax(q0)
print(qC)
# compute P from initial q
P.interpolate(P_expr)
print_minmax(P)
# compute initial w
w.interpolate(w_expr)

plot_field_latlon(q0, 'initial_q')
plot_field_latlon(P, 'initial_P')
plot_field_latlon(w, 'initial_w')

# =======================================================================
# Now we can timestep!
t = 0
tmax = 1000 * dt
not_steady = True   # flag to indicate that we have not yet reached a
                    # steady state
not_blowing_up = True
first_step = True
count = 0
tol = 1e-3          # tolerance with which to compute steady state

# timeloop
while not_steady and not_blowing_up and t < tmax:
    count += 1
    t += dt
    # update values in w
    w.interpolate(w_expr)
    # compute h
    h_solver.solve()
    # compute u
    u_solver.solve()
    # update functions required to compute q
    qA.interpolate(qA_expr)
    E.interpolate(E_expr)
    P.interpolate(P_expr)
    # compute q
    q_solver.solve()

    print(f"at time {t}, change in h is: {norm(h-h0)}")
    # print(assemble(w*dx))
    not_steady = norm(h-h0) > tol
    if first_step:
        initial_diff = norm(h-h0)
        first_step = False
    not_blowing_up = norm(h-h0) < 10 * initial_diff

    # update fields
    h0.assign(h)
    u0.assign(u)
    q0.assign(q)

    # =======================================================================
    # Now let's plot the final fields.
    if count % 50 == 0 or not not_blowing_up:
        plot_field_latlon(q, f'q_{t}')
        plot_field_latlon(h, f'h_{t}')
        plot_field_latlon(u, f'u_{t}')
        plot_u_components(u, f'u_{t}')

# =======================================================================
# check that q has changed
q_diff = Function(q0.function_space(), name='qdiff').assign(q-q_init)
print_minmax(q_diff)
plot_field_latlon(q_diff, "dq")

