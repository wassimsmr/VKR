import numpy as np


class BaseSolver:
    """Base class for all numerical solvers."""

    def __init__(self, equation, domain, boundary_conditions, **kwargs):
        """
        Initialize the base solver.

        Args:
            equation (callable): Function that defines the differential equation
            domain (tuple): Tuple (t_min, t_max) for the domain
            boundary_conditions (dict): Dictionary of boundary conditions
            **kwargs: Additional solver-specific parameters
        """
        self.equation = equation
        self.domain = domain
        self.boundary_conditions = boundary_conditions
        self.solution = None

    def solve(self):
        """Solve the differential equation."""
        raise NotImplementedError("Subclasses must implement solve method")

    def evaluate(self, t):
        """
        Evaluate the solution at given points.

        Args:
            t (numpy.ndarray): Points to evaluate the solution at

        Returns:
            numpy.ndarray: Solution values at the given points
        """
        raise NotImplementedError("Subclasses must implement evaluate method")

    def get_error(self, exact_solution, t=None):
        """
        Compute the error between the numerical and exact solutions.

        Args:
            exact_solution (callable): Exact solution function
            t (numpy.ndarray, optional): Points to evaluate the error at

        Returns:
            float: L2 norm of the error
        """
        if t is None:
            t = np.linspace(self.domain[0], self.domain[1], 1000)

        numerical_solution = self.evaluate(t)
        exact_values = np.array([exact_solution(ti) for ti in t])

        # Compute L2 norm of the error
        l2_error = np.sqrt(np.mean((numerical_solution - exact_values) ** 2))

        return l2_error

    def get_config(self):
        """Get solver configuration."""
        return {
            'type': self.__class__.__name__,
            'domain': self.domain,
            'boundary_conditions': self.boundary_conditions
        }