import numpy as np
from Solvers.base import BaseSolver


class RungeKuttaSolver(BaseSolver):
    """Advanced Runge-Kutta methods for solving ODEs."""

    def __init__(self, equation, domain, boundary_conditions, method='rk4', num_points=1000, adaptive=False,
                 tolerance=1e-6):
        """
        Initialize the Runge-Kutta solver.

        Args:
            equation (callable): Function that defines the differential equation
            domain (tuple): Tuple (t_min, t_max) for the domain
            boundary_conditions (dict): Dictionary of boundary conditions
            method (str): Runge-Kutta method ('rk1', 'rk2', 'rk3', 'rk4', 'rk45', 'rkf')
            num_points (int): Number of discretization points (for non-adaptive methods)
            adaptive (bool): Whether to use adaptive step size control
            tolerance (float): Error tolerance for adaptive step size control
        """
        super(RungeKuttaSolver, self).__init__(equation, domain, boundary_conditions)

        self.method = method
        self.num_points = num_points
        self.adaptive = adaptive
        self.tolerance = tolerance

        # Initialize solution arrays
        if not adaptive:
            self.t = np.linspace(domain[0], domain[1], num_points)
            self.dt = (domain[1] - domain[0]) / (num_points - 1)
        else:
            # For adaptive methods, arrays will be built during solving
            self.t = None
            self.dt = None

        self.solution = None

    def _rk1_step(self, t, y, dt):
        """Euler method (RK1) step."""
        k1 = self.equation(t, y)
        return y + dt * k1

    def _rk2_step(self, t, y, dt):
        """Midpoint method (RK2) step."""
        k1 = self.equation(t, y)
        k2 = self.equation(t + 0.5 * dt, y + 0.5 * dt * k1)
        return y + dt * k2

    def _rk3_step(self, t, y, dt):
        """Kutta's third-order method step."""
        k1 = self.equation(t, y)
        k2 = self.equation(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = self.equation(t + dt, y - dt * k1 + 2 * dt * k2)
        return y + dt * (k1 + 4 * k2 + k3) / 6

    def _rk4_step(self, t, y, dt):
        """Classical fourth-order Runge-Kutta method step."""
        k1 = self.equation(t, y)
        k2 = self.equation(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = self.equation(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = self.equation(t + dt, y + dt * k3)
        return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def _rkf45_step(self, t, y, dt):
        """
        Runge-Kutta-Fehlberg (RKF45) method step with adaptive step size control.
        Provides 4th order approximation with 5th order error estimation.

        Returns:
            tuple: (new_y, error_estimate, new_dt)
        """
        # RKF45 coefficients
        a2, a3, a4, a5, a6 = 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2
        b21 = 1 / 4
        b31, b32 = 3 / 32, 9 / 32
        b41, b42, b43 = 1932 / 2197, -7200 / 2197, 7296 / 2197
        b51, b52, b53, b54 = 439 / 216, -8, 3680 / 513, -845 / 4104
        b61, b62, b63, b64, b65 = -8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40

        # 4th order Runge-Kutta coefficients
        c1, c3, c4, c5 = 25 / 216, 1408 / 2565, 2197 / 4104, -1 / 5

        # 5th order Runge-Kutta coefficients for error estimation
        ce1, ce3, ce4, ce5, ce6 = 16 / 135, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55

        # Compute k values
        k1 = self.equation(t, y)
        k2 = self.equation(t + a2 * dt, y + dt * (b21 * k1))
        k3 = self.equation(t + a3 * dt, y + dt * (b31 * k1 + b32 * k2))
        k4 = self.equation(t + a4 * dt, y + dt * (b41 * k1 + b42 * k2 + b43 * k3))
        k5 = self.equation(t + a5 * dt, y + dt * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4))
        k6 = self.equation(t + a6 * dt, y + dt * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5))

        # 4th order approximation
        y4 = y + dt * (c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5)

        # 5th order approximation
        y5 = y + dt * (ce1 * k1 + ce3 * k3 + ce4 * k4 + ce5 * k5 + ce6 * k6)

        # Error estimation
        error = np.abs(y5 - y4)

        # Adaptive step size control
        if error > 0:
            # Scale factor with safety factor 0.9
            scale = 0.9 * (self.tolerance / error) ** 0.2
            # Limit scale to reasonable range
            scale = min(max(scale, 0.1), 4.0)
            new_dt = dt * scale
        else:
            new_dt = dt * 2.0

        return y5, error, new_dt

    def _dormand_prince_step(self, t, y, dt):
        """
        Dormand-Prince (DOPRI) method step with adaptive step size control.
        Higher order variant of RKF45 with better error estimates.

        Returns:
            tuple: (new_y, error_estimate, new_dt)
        """
        # DOPRI coefficients
        a = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])

        # b coefficients for intermediate steps
        b = np.zeros((7, 6))
        b[1, 0] = 1 / 5
        b[2, 0:2] = [3 / 40, 9 / 40]
        b[3, 0:3] = [44 / 45, -56 / 15, 32 / 9]
        b[4, 0:4] = [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]
        b[5, 0:5] = [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]
        b[6, 0:6] = [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]

        # 5th order solution coefficients
        c5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])

        # 4th order solution coefficients for error estimation
        c4 = np.array([5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40])

        # Compute k values
        k = np.zeros((7, y.shape[0]))
        k[0] = self.equation(t, y)

        for i in range(1, 7):
            yi = y.copy()
            for j in range(i):
                yi += dt * b[i, j] * k[j]
            k[i] = self.equation(t + a[i] * dt, yi)

        # 5th order approximation
        y5 = y.copy()
        for i in range(7):
            y5 += dt * c5[i] * k[i]

        # 4th order approximation for error estimation
        y4 = y.copy()
        for i in range(7):
            y4 += dt * c4[i] * k[i]

        # Error estimation
        error = np.max(np.abs(y5 - y4))

        # Adaptive step size control
        if error > 0:
            # Scale factor with safety factor 0.9
            scale = 0.9 * (self.tolerance / error) ** 0.2
            # Limit scale to reasonable range
            scale = min(max(scale, 0.1), 4.0)
            new_dt = dt * scale
        else:
            new_dt = dt * 2.0

        return y5, error, new_dt

    def solve_adaptive(self):
        """
        Solve the differential equation using adaptive step size control.

        Returns:
            tuple: (t, y) arrays of solution points
        """
        # Initial conditions
        t_start, t_end = self.domain
        y = np.array([self.boundary_conditions.get('initial_value', 0.0)])
        t = np.array([t_start])

        # Initial step size
        dt = (t_end - t_start) / 100.0  # Start with a reasonable step size

        # Integration loop
        current_t = t_start

        while current_t < t_end:
            # Make sure we don't overshoot the end
            if current_t + dt > t_end:
                dt = t_end - current_t

            # Compute step based on selected method
            if self.method == 'rkf45':
                new_y, error, new_dt = self._rkf45_step(current_t, y[-1], dt)
            elif self.method == 'dopri':
                new_y, error, new_dt = self._dormand_prince_step(current_t, y[-1], dt)
            else:
                raise ValueError(f"Adaptive step size not supported for method {self.method}")

            # Update solution
            current_t += dt
            t = np.append(t, current_t)
            y = np.append(y, [new_y], axis=0)

            # Update step size for next iteration
            dt = new_dt

        self.t = t
        self.solution = y
        return t, y

    def solve_fixed(self):
        """
        Solve the differential equation using fixed step size.

        Returns:
            numpy.ndarray: Solution array
        """
        # Initialize solution array
        y = np.zeros(self.num_points)

        # Set initial condition
        y[0] = self.boundary_conditions.get('initial_value', 0.0)

        # Integration loop
        for i in range(self.num_points - 1):
            t_i = self.t[i]
            y_i = y[i]

            # Select method
            if self.method == 'rk1':
                y[i + 1] = self._rk1_step(t_i, y_i, self.dt)
            elif self.method == 'rk2':
                y[i + 1] = self._rk2_step(t_i, y_i, self.dt)
            elif self.method == 'rk3':
                y[i + 1] = self._rk3_step(t_i, y_i, self.dt)
            elif self.method == 'rk4':
                y[i + 1] = self._rk4_step(t_i, y_i, self.dt)
            else:
                raise ValueError(f"Unknown fixed-step method: {self.method}")

        self.solution = y
        return y

    def solve(self):
        """
        Solve the differential equation using the selected Runge-Kutta method.

        Returns:
            numpy.ndarray: Solution array
        """
        if self.adaptive:
            return self.solve_adaptive()
        else:
            return self.solve_fixed()

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
        config = super(RungeKuttaSolver, self).get_config()
        config.update({
            'method': self.method,
            'num_points': self.num_points,
            'adaptive': self.adaptive,
            'tolerance': self.tolerance
        })
        return config