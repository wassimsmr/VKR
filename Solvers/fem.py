import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from Solvers.base import BaseSolver


class FEM(BaseSolver):
    """Finite Element Method solver."""

    def __init__(self, equation, domain, boundary_conditions, num_elements=100):
        """
        Initialize the FEM solver.

        Args:
            equation (callable): Function that defines the differential equation (must be linear)
            domain (tuple): Tuple (t_min, t_max) for the domain
            boundary_conditions (dict): Dictionary of boundary conditions
            num_elements (int): Number of finite elements
        """
        super(FEM, self).__init__(equation, domain, boundary_conditions)

        self.num_elements = num_elements
        self.num_nodes = num_elements + 1
        self.nodes = np.linspace(domain[0], domain[1], self.num_nodes)
        self.element_size = (domain[1] - domain[0]) / num_elements
        self.solution = None

    def assemble_system(self):
        """
        Assemble the FEM system matrices.

        Returns:
            tuple: (A, b) where A is the system matrix and b is the right-hand side
        """
        # For second-order ODE: -d/dx(p(x)*du/dx) + q(x)*u = f(x)
        # Assuming p(x) = 1, q(x) = 0, f(x) is the right-hand side

        # Initialize sparse matrix and right-hand side vector
        A = sparse.lil_matrix((self.num_nodes, self.num_nodes))
        b = np.zeros(self.num_nodes)

        # Assemble element matrices and add to global system
        for i in range(self.num_elements):
            x_left = self.nodes[i]
            x_right = self.nodes[i + 1]
            h = x_right - x_left

            # Element stiffness matrix (for -d/dx(du/dx))
            element_A = np.array([[1, -1], [-1, 1]]) / h

            # Add to global matrix
            A[i, i] += element_A[0, 0]
            A[i, i + 1] += element_A[0, 1]
            A[i + 1, i] += element_A[1, 0]
            A[i + 1, i + 1] += element_A[1, 1]

            # Element load vector (for right-hand side)
            # For simplicity, we use a constant approximation of f(x) in each element
            f_mid = self.equation((x_left + x_right) / 2, 0)  # Just using midpoint value
            element_b = np.array([f_mid, f_mid]) * h / 2

            # Add to global right-hand side
            b[i] += element_b[0]
            b[i + 1] += element_b[1]

        return A.tocsr(), b

    def apply_boundary_conditions(self, A, b):
        """
        Apply boundary conditions to the system.

        Args:
            A (scipy.sparse.csr_matrix): System matrix
            b (numpy.ndarray): Right-hand side vector

        Returns:
            tuple: Modified (A, b)
        """
        # Apply Dirichlet boundary conditions
        if 'left_value' in self.boundary_conditions:
            value = self.boundary_conditions['left_value']

            # Modify right-hand side
            b = b - A[:, 0] * value

            # Modify matrix (set row and column to zero, diagonal to 1)
            A = A.tolil()
            A[0, :] = 0
            A[0, 0] = 1
            A = A.tocsr()

            # Set right-hand side
            b[0] = value

        if 'right_value' in self.boundary_conditions:
            value = self.boundary_conditions['right_value']

            # Modify right-hand side
            b = b - A[:, -1] * value

            # Modify matrix
            A = A.tolil()
            A[-1, :] = 0
            A[-1, -1] = 1
            A = A.tocsr()

            # Set right-hand side
            b[-1] = value

        return A, b

    def solve(self):
        """
        Solve the differential equation using FEM.

        Returns:
            numpy.ndarray: Solution array
        """
        # Assemble system
        A, b = self.assemble_system()

        # Apply boundary conditions
        A, b = self.apply_boundary_conditions(A, b)

        # Solve the system
        self.solution = spsolve(A, b)

        return self.solution

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
        return np.interp(t, self.nodes, self.solution)

    def get_config(self):
        """Get solver configuration."""
        config = super().get_config()
        config.update({
            'num_elements': self.num_elements,
            'element_size': self.element_size
        })
        return config