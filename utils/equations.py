import numpy as np
import torch


class DifferentialEquation:
    """Base class for differential equations."""

    def __init__(self, name="Generic Differential Equation"):
        """
        Initialize the differential equation.

        Args:
            name (str): Name of the differential equation
        """
        self.name = name

    def __call__(self, t, y):
        """
        Evaluate the differential equation.

        Args:
            t (float or numpy.ndarray or torch.Tensor): Time point(s)
            y (float or numpy.ndarray or torch.Tensor): Solution value(s)

        Returns:
            float or numpy.ndarray or torch.Tensor: Result of the differential equation
        """
        raise NotImplementedError("Subclasses must implement __call__ method")

    def exact_solution(self, t):
        """
        Evaluate the exact solution if available.

        Args:
            t (float or numpy.ndarray): Time point(s)

        Returns:
            float or numpy.ndarray: Exact solution value(s)
        """
        raise NotImplementedError("Subclasses must implement exact_solution method")


class LinearODE(DifferentialEquation):
    """Linear first-order ODE: dy/dt = a*y + b."""

    def __init__(self, a=1.0, b=0.0, initial_value=1.0):
        """
        Initialize the linear ODE.

        Args:
            a (float): Coefficient of y
            b (float): Constant term
            initial_value (float): Initial value y(0)
        """
        super(LinearODE, self).__init__(name=f"Linear ODE: dy/dt = {a}*y + {b}")
        self.a = a
        self.b = b
        self.initial_value = initial_value

    def __call__(self, t, y):
        """
        Evaluate the differential equation.

        Args:
            t (float or numpy.ndarray or torch.Tensor): Time point(s)
            y (float or numpy.ndarray or torch.Tensor): Solution value(s)

        Returns:
            float or numpy.ndarray or torch.Tensor: Result of the differential equation
        """
        return self.a * y + self.b

    def exact_solution(self, t):
        """
        Evaluate the exact solution.

        Args:
            t (float or numpy.ndarray): Time point(s)

        Returns:
            float or numpy.ndarray: Exact solution value(s)
        """
        if self.a == 0:
            return self.initial_value + self.b * t
        else:
            return self.initial_value * np.exp(self.a * t) + (self.b / self.a) * (np.exp(self.a * t) - 1)


class HarmonicOscillator(DifferentialEquation):
    """Simple harmonic oscillator: d²y/dt² + ω²*y = 0."""

    def __init__(self, omega=1.0, initial_position=1.0, initial_velocity=0.0):
        """
        Initialize the harmonic oscillator.

        Args:
            omega (float): Angular frequency
            initial_position (float): Initial position y(0)
            initial_velocity (float): Initial velocity y'(0)
        """
        super(HarmonicOscillator, self).__init__(
            name=f"Harmonic Oscillator: d²y/dt² + {omega}²*y = 0"
        )
        self.omega = omega
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity

        # Compute amplitude and phase
        self.amplitude = np.sqrt(initial_position ** 2 + (initial_velocity / omega) ** 2)
        self.phase = np.arctan2(initial_velocity / omega, initial_position)

    def __call__(self, t, y, dy_dt=None):
        """
        Evaluate the second-order differential equation as a system of first-order ODEs.

        For a second-order ODE, we need to convert it to a system of first-order ODEs:
        Let y1 = y and y2 = dy/dt, then:
        dy1/dt = y2
        dy2/dt = -ω²*y1

        Args:
            t (float or numpy.ndarray or torch.Tensor): Time point(s)
            y (float or numpy.ndarray or torch.Tensor): Position value(s)
            dy_dt (float or numpy.ndarray or torch.Tensor, optional): Velocity value(s)

        Returns:
            tuple or float or numpy.ndarray or torch.Tensor: Result of the differential equation
        """
        if dy_dt is None:
            # Return the second derivative for PINNs
            return -self.omega ** 2 * y
        else:
            # Return the system of first-order ODEs for classical solvers
            return np.array([dy_dt, -self.omega ** 2 * y])

    def exact_solution(self, t):
        """
        Evaluate the exact solution.

        Args:
            t (float or numpy.ndarray): Time point(s)

        Returns:
            float or numpy.ndarray: Exact solution value(s)
        """
        return self.amplitude * np.cos(self.omega * t + self.phase)


class NonlinearODE(DifferentialEquation):
    """Nonlinear ODE: dy/dt = f(t, y)."""

    def __init__(self, f, name="Nonlinear ODE", exact_sol=None):
        """
        Initialize the nonlinear ODE.

        Args:
            f (callable): Function f(t, y) that defines the ODE dy/dt = f(t, y)
            name (str): Name of the differential equation
            exact_sol (callable, optional): Exact solution function
        """
        super(NonlinearODE, self).__init__(name=name)
        self.f = f
        self.exact_sol = exact_sol

    def __call__(self, t, y):
        """
        Evaluate the differential equation.

        Args:
            t (float or numpy.ndarray or torch.Tensor): Time point(s)
            y (float or numpy.ndarray or torch.Tensor): Solution value(s)

        Returns:
            float or numpy.ndarray or torch.Tensor: Result of the differential equation
        """
        return self.f(t, y)

    def exact_solution(self, t):
        """
        Evaluate the exact solution if available.

        Args:
            t (float or numpy.ndarray): Time point(s)

        Returns:
            float or numpy.ndarray: Exact solution value(s)
        """
        if self.exact_sol is None:
            raise NotImplementedError("Exact solution not available for this nonlinear ODE")
        return self.exact_sol(t)


class SystemODE(DifferentialEquation):
    """System of ODEs: dy/dt = f(t, y), where y is a vector."""

    def __init__(self, f, dimension=2, name="System of ODEs", exact_sol=None):
        """
        Initialize the system of ODEs.

        Args:
            f (callable): Function f(t, y) that defines the system: dy/dt = f(t, y)
            dimension (int): Dimension of the system (number of variables)
            name (str): Name of the differential equation
            exact_sol (callable, optional): Exact solution function
        """
        super(SystemODE, self).__init__(name=name)
        self.f = f
        self.dimension = dimension
        self.exact_sol = exact_sol

    def __call__(self, t, y):
        """
        Evaluate the system of differential equations.

        Args:
            t (float or numpy.ndarray or torch.Tensor): Time point(s)
            y (float or numpy.ndarray or torch.Tensor): Solution vector(s)

        Returns:
            float or numpy.ndarray or torch.Tensor: Result of the differential equation system
        """
        return self.f(t, y)

    def exact_solution(self, t):
        """
        Evaluate the exact solution if available.

        Args:
            t (float or numpy.ndarray): Time point(s)

        Returns:
            float or numpy.ndarray: Exact solution value(s)
        """
        if self.exact_sol is None:
            raise NotImplementedError("Exact solution not available for this system of ODEs")
        return self.exact_sol(t)


class AdvectionDiffusionPDE(DifferentialEquation):
    """
    One-dimensional advection-diffusion PDE:
    ∂u/∂t + v ∂u/∂x = D ∂²u/∂x²
    """

    def __init__(self, velocity=1.0, diffusion=0.01, initial_condition=None, name=None):
        """
        Initialize the advection-diffusion PDE.

        Args:
            velocity (float): Advection velocity
            diffusion (float): Diffusion coefficient
            initial_condition (callable, optional): Initial condition function u(x, 0)
            name (str, optional): Name of the PDE
        """
        if name is None:
            name = f"Advection-Diffusion PDE: ∂u/∂t + {velocity} ∂u/∂x = {diffusion} ∂²u/∂x²"
        super(AdvectionDiffusionPDE, self).__init__(name=name)

        self.velocity = velocity
        self.diffusion = diffusion

        # Default initial condition: Gaussian pulse
        if initial_condition is None:
            self.initial_condition = lambda x: np.exp(-(x - 0.5) ** 2 / 0.1)
        else:
            self.initial_condition = initial_condition

    def __call__(self, t, x, u, u_x=None, u_xx=None):
        """
        Evaluate the PDE residual.

        Args:
            t (float or numpy.ndarray or torch.Tensor): Time
            x (float or numpy.ndarray or torch.Tensor): Space
            u (float or numpy.ndarray or torch.Tensor): Solution value
            u_x (float or numpy.ndarray or torch.Tensor, optional): First spatial derivative
            u_xx (float or numpy.ndarray or torch.Tensor, optional): Second spatial derivative

        Returns:
            float or numpy.ndarray or torch.Tensor: PDE residual
        """
        if u_x is None or u_xx is None:
            raise ValueError("Spatial derivatives u_x and u_xx must be provided")

        # PDE residual: ∂u/∂t + v ∂u/∂x - D ∂²u/∂x² = 0
        return self.velocity * u_x - self.diffusion * u_xx

    def exact_solution(self, t, x):
        """
        Evaluate the exact solution for a Gaussian initial condition.

        This is only valid for the infinite domain case with a Gaussian initial condition.

        Args:
            t (float or numpy.ndarray): Time point(s)
            x (float or numpy.ndarray): Space point(s)

        Returns:
            float or numpy.ndarray: Exact solution value(s)
        """
        # For a Gaussian initial pulse in an infinite domain
        if not callable(self.initial_condition):
            raise NotImplementedError("Exact solution available only for Gaussian initial condition")

        # Create meshgrid if needed
        if isinstance(t, np.ndarray) and isinstance(x, np.ndarray):
            if len(t.shape) == 1 and len(x.shape) == 1:
                t_mesh, x_mesh = np.meshgrid(t, x, indexing='ij')
                t = t_mesh
                x = x_mesh

        # Initial Gaussian pulse at x_0 = 0.5 with width sigma = 0.1
        x_0 = 0.5
        sigma = 0.1

        # Compute the solution: Gaussian that advects and diffuses
        sigma_t = np.sqrt(sigma ** 2 + 2 * self.diffusion * t)
        x_t = x_0 + self.velocity * t
        u = (sigma / sigma_t) * np.exp(-(x - x_t) ** 2 / (2 * sigma_t ** 2))

        return u


class WaveEquationPDE(DifferentialEquation):
    """
    One-dimensional wave equation PDE:
    ∂²u/∂t² = c² ∂²u/∂x²
    """

    def __init__(self, wave_speed=1.0, initial_displacement=None, initial_velocity=None, name=None):
        """
        Initialize the wave equation PDE.

        Args:
            wave_speed (float): Wave propagation speed
            initial_displacement (callable, optional): Initial displacement function u(x, 0)
            initial_velocity (callable, optional): Initial velocity function ∂u/∂t(x, 0)
            name (str, optional): Name of the PDE
        """
        if name is None:
            name = f"Wave Equation PDE: ∂²u/∂t² = {wave_speed}² ∂²u/∂x²"
        super(WaveEquationPDE, self).__init__(name=name)

        self.wave_speed = wave_speed

        # Default initial conditions: Gaussian pulse with zero initial velocity
        if initial_displacement is None:
            self.initial_displacement = lambda x: np.exp(-(x - 0.5) ** 2 / 0.1)
        else:
            self.initial_displacement = initial_displacement

        if initial_velocity is None:
            self.initial_velocity = lambda x: np.zeros_like(x)
        else:
            self.initial_velocity = initial_velocity

    def __call__(self, t, x, u, u_t=None, u_x=None, u_xx=None, u_tt=None):
        """
        Evaluate the PDE residual.

        Args:
            t (float or numpy.ndarray or torch.Tensor): Time
            x (float or numpy.ndarray or torch.Tensor): Space
            u (float or numpy.ndarray or torch.Tensor): Solution value
            u_t (float or numpy.ndarray or torch.Tensor, optional): First time derivative
            u_x (float or numpy.ndarray or torch.Tensor, optional): First spatial derivative
            u_xx (float or numpy.ndarray or torch.Tensor, optional): Second spatial derivative
            u_tt (float or numpy.ndarray or torch.Tensor, optional): Second time derivative

        Returns:
            float or numpy.ndarray or torch.Tensor: PDE residual
        """
        if u_tt is None or u_xx is None:
            raise ValueError("Second derivatives u_tt and u_xx must be provided")

        # PDE residual: ∂²u/∂t² - c² ∂²u/∂x² = 0
        return u_tt - self.wave_speed ** 2 * u_xx

    def exact_solution(self, t, x):
        """
        Evaluate the exact solution using D'Alembert's formula.

        This is valid only for an infinite domain with specified initial conditions.

        Args:
            t (float or numpy.ndarray): Time point(s)
            x (float or numpy.ndarray): Space point(s)

        Returns:
            float or numpy.ndarray: Exact solution value(s)
        """
        # D'Alembert's formula: u(x, t) = 0.5[f(x + ct) + f(x - ct)] + (1/2c)∫(x-ct)^(x+ct) g(s) ds
        # where f(x) is initial displacement and g(x) is initial velocity

        # For simplicity, we only implement the case with zero initial velocity
        if not callable(self.initial_velocity) or not callable(self.initial_displacement):
            raise NotImplementedError("Exact solution requires callable initial conditions")

        # Create meshgrid if needed
        if isinstance(t, np.ndarray) and isinstance(x, np.ndarray):
            if len(t.shape) == 1 and len(x.shape) == 1:
                t_mesh, x_mesh = np.meshgrid(t, x, indexing='ij')
                t = t_mesh
                x = x_mesh

        # Compute left-traveling and right-traveling waves
        x_plus_ct = x + self.wave_speed * t
        x_minus_ct = x - self.wave_speed * t

        # D'Alembert's formula for the case of zero initial velocity
        u = 0.5 * (self.initial_displacement(x_plus_ct) + self.initial_displacement(x_minus_ct))

        # For non-zero initial velocity, we would need to add an integral term
        if callable(self.initial_velocity) and np.any(self.initial_velocity(np.array([0.0])) != 0):
            # This is a simplified approximation of the integral term
            dx = 0.01  # Integration step size
            for s in np.arange(x_minus_ct, x_plus_ct, dx):
                u += 0.5 * dx / self.wave_speed * self.initial_velocity(s)

        return u


class ParametricODE(DifferentialEquation):
    """Parametric ODE with customizable form."""

    def __init__(self, form, params, initial_conditions, exact_solution_func=None, name=None):
        """
        Initialize a parametric ODE.

        Args:
            form (str): String representation of the ODE (e.g., 'dy/dt = a*y + b')
            params (dict): Dictionary of parameters (e.g., {'a': 1.0, 'b': 2.0})
            initial_conditions (dict): Dictionary of initial conditions
            exact_solution_func (callable, optional): Function for the exact solution
            name (str, optional): Name of the equation
        """
        if name is None:
            name = form
        super(ParametricODE, self).__init__(name=name)

        self.form = form
        self.params = params
        self.initial_conditions = initial_conditions
        self.exact_solution_func = exact_solution_func

        # Create lambda function for ODE evaluation based on form and params

        if form == 'dy/dt = a*y + b':
            self.a = params.get('a', 0.0)
            self.b = params.get('b', 0.0)
            self.func = lambda t, y: self.a * y + self.b
        elif form == 'dy/dt = a*y^2 + b*y + c':
            self.a = params.get('a', 0.0)
            self.b = params.get('b', 0.0)
            self.c = params.get('c', 0.0)
            self.func = lambda t, y: self.a * y ** 2 + self.b * y + self.c
        else:
            raise ValueError(f"Unsupported ODE form: {form}")

    def __call__(self, t, y):
        """Evaluate the ODE."""
        return self.func(t, y)

    def exact_solution(self, t):
        """Evaluate the exact solution if available."""
        if self.exact_solution_func is None:
            raise NotImplementedError("Exact solution not available for this ODE")
        return self.exact_solution_func(t, self.params, self.initial_conditions)


class PartialDifferentialEquation(DifferentialEquation):
    """Base class for partial differential equations."""

    def __init__(self, name="Generic PDE"):
        """
        Initialize the partial differential equation.

        Args:
            name (str): Name of the PDE
        """
        super(PartialDifferentialEquation, self).__init__(name=name)

    def __call__(self, x, y, u, u_x=None, u_y=None, u_xx=None, u_yy=None, u_xy=None):
        """
        Evaluate the PDE residual.

        Args:
            x (torch.Tensor): x-coordinate points
            y (torch.Tensor): y-coordinate points
            u (torch.Tensor): Function value at (x,y)
            u_x (torch.Tensor, optional): First derivative with respect to x
            u_y (torch.Tensor, optional): First derivative with respect to y
            u_xx (torch.Tensor, optional): Second derivative with respect to x
            u_yy (torch.Tensor, optional): Second derivative with respect to y
            u_xy (torch.Tensor, optional): Mixed derivative

        Returns:
            torch.Tensor: Residual of the PDE
        """
        raise NotImplementedError("Subclasses must implement __call__ method")

    def boundary_condition(self, x, y):
        """
        Evaluate the boundary condition.

        Args:
            x (torch.Tensor): x-coordinate points
            y (torch.Tensor): y-coordinate points

        Returns:
            torch.Tensor: Boundary condition values
        """
        raise NotImplementedError("Subclasses must implement boundary_condition method")

class PoissonEquation(PartialDifferentialEquation):
        """
        Poisson Equation: ∇²u = f(x,y)
        or u_xx + u_yy = f(x,y)
        """

        def __init__(self, source_term, name=None):
            """
            Initialize the Poisson equation.

            Args:
                source_term (callable): Right-hand side function f(x,y)
                name (str, optional): Name of the PDE
            """
            if name is None:
                name = "Poisson Equation: ∇²u = f(x,y)"
            super(PoissonEquation, self).__init__(name=name)

            self.source_term = source_term

        def __call__(self, x, y, u, u_x=None, u_y=None, u_xx=None, u_yy=None, u_xy=None):
            """
            Evaluate the PDE residual.

            Args:
                x (torch.Tensor): x-coordinate points
                y (torch.Tensor): y-coordinate points
                u (torch.Tensor): Function value at (x,y)
                u_x, u_y, u_xx, u_yy, u_xy: Derivatives (computed if not provided)

            Returns:
                torch.Tensor: Residual of the PDE
            """
            # Compute derivatives if not provided
            if u_xx is None or u_yy is None:
                if u_x is None:
                    u_x = torch.autograd.grad(
                        u, x, torch.ones_like(u), create_graph=True, retain_graph=True
                    )[0]

                if u_y is None:
                    u_y = torch.autograd.grad(
                        u, y, torch.ones_like(u), create_graph=True, retain_graph=True
                    )[0]

                u_xx = torch.autograd.grad(
                    u_x, x, torch.ones_like(u_x), create_graph=True, retain_graph=True
                )[0]

                u_yy = torch.autograd.grad(
                    u_y, y, torch.ones_like(u_y), create_graph=True, retain_graph=True
                )[0]

            # Compute source term
            f = self.source_term(x, y)

            # PDE residual: u_xx + u_yy - f(x,y) = 0
            return u_xx + u_yy - f

        def boundary_condition(self, x, y, boundary_type="dirichlet", boundary_value=0.0):
            """
            Evaluate the boundary condition.

            Args:
                x (torch.Tensor): x-coordinate points
                y (torch.Tensor): y-coordinate points
                boundary_type (str): Type of boundary condition
                boundary_value (float or callable): Boundary value

            Returns:
                float or callable: Boundary condition value
            """
            if callable(boundary_value):
                return boundary_value(x, y)
            else:
                return boundary_value
