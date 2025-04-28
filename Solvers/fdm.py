import numpy as np
from Solvers.base import BaseSolver


class FDM(BaseSolver):
    """Finite Difference Method solver."""

    def __init__(self, equation, domain, boundary_conditions, num_points=100, time_step=None):
        """
        Initialize the FDM solver.

        Args:
            equation (callable): Function that defines the differential equation
            domain (tuple): Tuple (t_min, t_max) for the domain
            boundary_conditions (dict): Dictionary of boundary conditions
            num_points (int): Number of discretization points
            time_step (float, optional): Time step size (if None, computed from domain and num_points)
        """
        super(FDM, self).__init__(equation, domain, boundary_conditions)

        self.num_points = num_points
        self.t = np.linspace(domain[0], domain[1], num_points)
        self.dt = (domain[1] - domain[0]) / (num_points - 1) if time_step is None else time_step
        self.solution = None

    def solve(self, method='euler'):
        """
        Solve the differential equation using FDM.

        Args:
            method (str): Integration method ('euler', 'rk2', 'rk4')

        Returns:
            numpy.ndarray: Solution array
        """
        # Initialize solution array
        y = np.zeros(self.num_points)

        # Set initial condition
        y0 = self.boundary_conditions.get('initial_value', 0)
        y[0] = y0

        # Integration methods
        if method == 'euler':
            # Forward Euler method
            for i in range(self.num_points - 1):
                t_i = self.t[i]
                y_i = y[i]
                y[i + 1] = y_i + self.dt * self.equation(t_i, y_i)

        elif method == 'rk2':
            # Runge-Kutta 2nd order (midpoint method)
            for i in range(self.num_points - 1):
                t_i = self.t[i]
                y_i = y[i]

                k1 = self.equation(t_i, y_i)
                k2 = self.equation(t_i + 0.5 * self.dt, y_i + 0.5 * self.dt * k1)

                y[i + 1] = y_i + self.dt * k2

        elif method == 'rk4':
            # Runge-Kutta 4th order
            for i in range(self.num_points - 1):
                t_i = self.t[i]
                y_i = y[i]

                k1 = self.equation(t_i, y_i)
                k2 = self.equation(t_i + 0.5 * self.dt, y_i + 0.5 * self.dt * k1)
                k3 = self.equation(t_i + 0.5 * self.dt, y_i + 0.5 * self.dt * k2)
                k4 = self.equation(t_i + self.dt, y_i + self.dt * k3)

                y[i + 1] = y_i + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        else:
            raise ValueError(f"Unknown method: {method}")

        self.solution = y
        return y

    def evaluate(self, t):
        """
        Evaluate the solution at given points.

        Args:
            t (numpy.ndarray): Points to evaluate the solution at

        Returns:
            numpy.ndarray: Solution values at the given points
        """
        if self.solution is None:
            raise ValueError("Solve the equation first using the solve() method")

        # Interpolate solution at the requested points
        return np.interp(t, self.t, self.solution)

    def get_config(self):
        """Get solver configuration."""
        config = super().get_config()
        config.update({
            'num_points': self.num_points,
            'dt': self.dt
        })
        return config