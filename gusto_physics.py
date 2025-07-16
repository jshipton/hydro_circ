from firedrake import (Function, conditional, dx, inner, dot, div, sqrt, exp,
                       Constant, FunctionSpace, assemble)
from firedrake.fml import subject
from ufl.domain import extract_unique_domain

#from gusto.core import EquationParameters
from gusto.core.labels import source_label
from gusto.physics import PhysicsParametrisation
import inspect


class EquationParameters(object):
    """A base configuration object for storing equation parameters."""

    mesh = None

    def __init__(self, mesh, **kwargs):
        """
        Args:
            mesh: for creating the real function space
            **kwargs: attributes and their values to be stored in the object.
        """
        self.mesh = mesh
        typecheck = lambda val: type(val) in [float, int, Constant]
        params = dict(inspect.getmembers(self, typecheck))
        params.update(kwargs.items())
        for name, value in params.items():
            self.__setattr__(name, value)

    def __setattr__(self, name, value):
        """
        Sets the model configuration attributes.

        When attributes are provided as floats or integers, these are converted
        to Firedrake :class:`Constant` objects, other than a handful of special
        integers.

        Args:
            name: the attribute's name.
            value: the value to provide to the attribute.

        Raises:
            AttributeError: if the :class:`Configuration` object does not have
                this attribute pre-defined.
        """
        if not hasattr(self, name):
            raise AttributeError("'%s' object has no attribute '%s'" % (type(self).__name__, name))

        # Almost all parameters should be functions on the real space
        # -- but there are some specific exceptions which should be
        # kept as integers
        if self.mesh is not None:
            # This check is required so that on instantiation we do
            # not hit this line while self.mesh is still None
            R = FunctionSpace(self.mesh, 'R', 0)
        if type(value) in [float, int, Constant]:
            object.__setattr__(self, name, Function(R, val=float(value)))
        else:
            object.__setattr__(self, name, value)


class HydroCircParameters(EquationParameters):
    Omega = 6.501e-6    # rotation rate of planet (rad s^-1)
    g = 10.9            # gravitational acceleration (m s^-2)
    H = 2500.           # depth of atmospheric boundary layer (mean
                        # fluid depth) (m)
    r = 3.6e-5          # linear coefficient of friction (N kg^-1 m^-1 s)
    # =======================================================================
    # Large scale moisture parameters
    q_ut = 0.004        # upper-tropospheric specific humidity (kg kg^-1)
    qW = q_ut           # tropically averaged descending moisture (kg
                        # kg^-1). q_ut for now, but should be somewhat
                        # higher probably.
    # =======================================================================
    # Parameters used to compute evaporation and precipitation
    rho0 = 1            # constant density of boundary layer air (kg m^-3)
    cH = 1e-3           # bulk coefficient (no units)
    L = 2.5e6           # latent heat of condensation (J kg^-1)
    Rw = 416            # specific gas constant for water vapour (J K^-1 kg^-1)
    e0 = 2300*exp(L/(Rw*293.))   # reference saturation vapour pressure (Pa)
    p0 = 1e5            # reference pressure (Pa)
    mB = 0.006029       # boundary layer overturning timescale (kg m^-2 s^-1)

    # =======================================================================
    # Parameters used to compute vertical velocity w
    Cp = 1005.          # specific heat capacity at constant pressure
                        # of dry air (J K^-1 kg^-1)
    dtheta = 50         # change in potential temperature across
                        # troposphere (K)
    Qcl = 100.          # net combined radiative-sensible cooling
                        # of free atmosphere (W m^-2)
    qC = 0.018          # critical specific humidity for
                        # initiation of convection (kg kg^-1)


def w(parameters, P):

    L = parameters.L
    Qcl = parameters.Qcl
    rho0 = parameters.rho0
    Cp = parameters.Cp
    dtheta = parameters.dtheta

    return (L * P - Qcl) / (rho0 * Cp * dtheta)


def precip(parameters, q):

    qC = parameters.qC
    mB = parameters.mB
    q_ut = parameters.q_ut

    return conditional(q > qC, mB * (q - q_ut), 0)


def evap(parameters, q, u, qs):

    cH = parameters.cH
    rho0 = parameters.rho0

    return conditional(
        qs > q,
        rho0 * cH * sqrt(dot(u, u)) * (qs - q),
        0)


def qW(q, P, w):

    descent_expr = conditional(w < 0, w, 0)
    ascent_expr = conditional(w >= 0, q * w - P, 0)
    return assemble(ascent_expr * dx) / assemble(descent_expr * dx)


class LinearFriction(PhysicsParametrisation):

    def __init__(self, equation):

        label_name = 'linear_friction'
        super().__init__(equation, label_name)

        r = self.parameters.r
        W = equation.function_space
        Vu = W.sub(0)
        test_u = equation.tests[0]
        self.u = Function(Vu)

        equation.residual += source_label(self.label(
            subject(inner(test_u, r * self.u) * dx, equation.X),
            self.evaluate
        ))

    def evaluate(self, x_in, dt, x_out=None):

        self.u.assign(x_in.subfunctions[0])


class VerticalVelocity(PhysicsParametrisation):

    def __init__(self, equation):

        label_name = 'vertical_velocity'
        super().__init__(equation, label_name)

        self.max_its = 100
        self.tol = 1e-8
        self.diff = 1e-1
        W = equation.function_space
        Vh = W.sub(1)
        test_h = equation.tests[1]
        self.q = Function(Vh)
        self.P = Function(Vh)
        self.w = Function(Vh)

        equation.residual += source_label(self.label(
            subject(test_h * self.w * dx, equation.X),
            self.evaluate
        ))

    def evaluate(self, x_in, dt, x_out=None):

        total_w = 1.
        nits = 0
        while abs(total_w) > self.tol and nits < self.max_its:

            self.q.assign(x_in.subfunctions[-1])
            self.P.interpolate(precip(self.parameters, self.q))
            self.w.assign(w(self.parameters, self.P))
            #print("w: ", self.w.dat.data.min(), self.w.dat.data.max())
            area = assemble(1*dx(domain=extract_unique_domain(self.w)))
            total_w = assemble(self.w * dx) / area
            if total_w > 0:
                self.parameters.qC += self.diff
            else:
                self.parameters.qC -= self.diff
            print(f"nits: {nits}, total_w: {total_w:.6f}, qC: {float(self.parameters.qC):.4f}, min w: {self.w.dat.data.min():.4f}, max w: {self.w.dat.data.max():.4f}")
            nits +=1
            

class Evaporation(PhysicsParametrisation):

    def __init__(self, equation, qs):

        label_name = 'evaporation'
        super().__init__(equation, label_name)

        rho0 = self.parameters.rho0
        H = self.parameters.H

        W = equation.function_space
        Vu = W.sub(0)
        self.u = Function(Vu)
        Vq = W.sub(-1)
        test_q = equation.tests[-1]
        self.q = Function(Vq)
        self.E = Function(Vq)
        self.qs = qs

        equation.residual -= source_label(self.label(
            subject(test_q * self.E / (rho0 * H) * dx, equation.X),
            self.evaluate
        ))

    def evaluate(self, x_in, dt, x_out=None):

        self.u.assign(x_in.subfunctions[0])
        self.q.assign(x_in.subfunctions[-1])
        self.E.interpolate(evap(self.parameters, self.q, self.u, self.qs))
        #print("E: ", self.E.dat.data.min(), self.E.dat.data.max())


class Precipitation(PhysicsParametrisation):

    def __init__(self, equation):

        label_name = 'precipitation'
        super().__init__(equation, label_name)

        rho0 = self.parameters.rho0
        H = self.parameters.H

        W = equation.function_space
        Vu = W.sub(0)
        self.u = Function(Vu)
        Vq = W.sub(-1)
        test_q = equation.tests[-1]
        self.q = Function(Vq)
        self.P = Function(Vq)

        equation.residual += source_label(self.label(
            subject(test_q * self.P / (rho0 * H) * dx, equation.X),
            self.evaluate
        ))

    def evaluate(self, x_in, dt, x_out=None):

        self.q.assign(x_in.subfunctions[-1])
        self.P.interpolate(precip(self.parameters, self.q))
        #print("P: ", self.P.dat.data.min(), self.P.dat.data.max())


class MoistureDescent(PhysicsParametrisation):

    def __init__(self, equation):

        label_name = 'moisture_descent'
        super().__init__(equation, label_name)

        W = equation.function_space
        Vu = W.sub(0)
        self.u = Function(Vu)
        Vq = W.sub(-1)
        test_q = equation.tests[-1]
        self.P = Function(Vq)
        self.w = Function(Vq)
        self.q = Function(Vq)
        self.qA = Function(Vq)
        self.descent_integral = Constant(0.)
        self.ascent_integral = Constant(0.)
        self.qW = Constant(0.)

        self.descent_expr = conditional(self.w < 0, self.w, 0)
        self.ascent_expr = conditional(self.w >= 0,
                                        self.q * self.w - self.P,
                                        0)
        self.qA_expr = conditional(self.w < 0, self.qW, 0)

        equation.residual -= source_label(self.label(
            subject(test_q * self.qA * div(self.u) * dx, equation.X),
            self.evaluate
        ))

    def evaluate(self, x_in, dt, x_out=None):

        self.q.assign(x_in.subfunctions[-1])
        self.P.interpolate(precip(self.parameters, self.q))
        self.w.assign(w(self.parameters, self.P))
        self.qW.assign(qW(self.q, self.P, self.w))
        self.qA.interpolate(self.qA_expr)
        self.u.assign(x_in.subfunctions[0])
        #print("qA: ", self.qA.dat.data.min(), self.qA.dat.data.max())
        #print("qW: ", float(self.qW))
