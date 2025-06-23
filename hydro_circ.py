import matplotlib.pyplot as plt
from matplotlib import cm

from firedrake import (SpatialCoordinate, TestFunction, TrialFunction, norm,
                       Function, dx, lhs, rhs, inner, LinearVariationalProblem,
                       LinearVariationalSolver, Constant, acos, exp, trisurf,
                       tricontourf, functionspaceimpl)
from gusto import *


def plot_field(field):
    # function to plot the field and show the plot - could be amended
    # to save a figure instead and other plotting commands can be added
    tsurf = trisurf(field)
    plt.colorbar(tsurf)
    plt.title(field.name())
    plt.show()


def plot_field_latlon(field):
    # DON'T USE ME!!
    # function to plot the field and show the plot - could be amended
    # to save a figure instead and other plotting commands can be added
    mesh_ll = get_flat_latlon_mesh(field.function_space().mesh())
    field_ll = Function(
        functionspaceimpl.WithGeometry.create(field.function_space(), mesh_ll),
        val=field.topological,
        name=field.name()
    )
    tsurf = tricontourf(field_ll)
    plt.colorbar(contourf)
    plt.title(field_ll.name())
    plt.show()


def print_minmax(field):
    # function to print the min and max of field in a nice way
    print(f"min and max of {field.name()}: {field.dat.data.min()}, {field.dat.data.max()}")


# Here are all the physical constants
R = 7160000.                 # radius of planet (m)
Omega = 6.501e-6             # rotation rate of planet
g = 10.9
H = 2500.
r = 3.6e-5
alpha = 500.
L = 2.5e6
Rw = 416
p0 = 1e5
e0 = 2300*exp(L/(Rw*293.))
q_ut = 0.004
mB = 0.006029
qW = q_ut
rho0 = 1
cH = 1e-3
Cp = 1005.
dtheta = 50
qC = 0.018
Qcl = 100.

# set up mesh, timestep and use the Gusto domain function to set up
# the finite element function spaces
mesh = GeneralCubedSphereMesh(radius=R, num_cells_per_edge_of_panel=16,
                              degree=2)
dt = 100
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

# Coriolis
fexpr = 2*Omega*sin(lat)
f = Function(Vcg).interpolate(fexpr)

# fake surface temperature field: a constant Tmin plus Gaussian
# perturbation centered on (lon_c, lat_c)
lon_c = 0
lat_c = 0
Tmin = 155
Tmax = 160

def d(lon1, lat1, lon2, lat2):
    # returns distance on sphere between (lon1, lat1) and (lon2, lat2)
    return acos(sin(lat1)*sin(lat2) + cos(lat1)*cos(lat2)*cos(lon1-lon2))

Ts = Function(Vdg, name='Ts')
Ts.interpolate(Tmin + Tmax * exp(-d(lon_c, lat_c, lon, lat)**2))

# saturation function
qs = 0.622 * e0 * exp(-L/(Rw*Ts)) / p0


# =======================================================================
# setup the h equation and solver
test_h = TestFunction(Vdg)
trial_h = TrialFunction(Vdg)
h0 = Function(Vdg)
h_eqn = test_h * trial_h * dx + dt * test_h * (-w + h0/alpha) * dx
h_lhs = lhs(h_eqn)
h_rhs = rhs(h_eqn)
h_prob = LinearVariationalProblem(h_lhs, h_rhs, h)
h_solver = LinearVariationalSolver(h_prob)


# =======================================================================
# setup the u equation and solver
test_u = TestFunction(Vu)
trial_u = TrialFunction(Vu)
u0 = Function(Vu)
u_eqn = inner(test_u, trial_u) * dx + dt * (
    inner(test_u, f*domain.perp(u) + r*u0) * dx - g * div(test_u) * h * dx
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
q_eqn = test_q * trial_q * dx + dt * test_q * ((q0 - qA) * div(u) - E + P) * dx
q_lhs = lhs(q_eqn)
q_rhs = rhs(q_eqn)
q_prob = LinearVariationalProblem(q_lhs, q_rhs, q)
q_solver = LinearVariationalSolver(q_prob)


# =======================================================================
# symbolic expressions for calculating various fields
w_expr = (L * P - Qcl) / (H * rho0 * Cp * dtheta)
qA_expr = conditional(w < 0, qW, 0)
E_expr = rho0 * cH * sqrt(dot(u, u)) * (qs - q)
P_expr = conditional(q > qC, mB * (q - q_ut), 0)


# =======================================================================
# initial conditions:
# initial water vapour, q, is 0.7 * saturation value
q.interpolate(0.7 * qs)
# compute P from initial q
P.interpolate(P_expr)
# compute initial w
w.interpolate(w_expr)

# =======================================================================
# Now we can timestep!
t = 0
tmax = 100 * dt
not_steady = True   # flag to indicate that we have not yet reached a
                    # steady state
tol = 1e-3          # tolerance with which to compute steady state

# timeloop
while not_steady and t < tmax:
    t += dt
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
    not_steady = norm(h-h0) > tol

    # update fields
    h0.assign(h)
    u0.assign(u)
    q0.assign(q)

plot_field_latlon(h)
plot_field(q)
plot_field(w)
plot_field(P)
plot_field(E)
