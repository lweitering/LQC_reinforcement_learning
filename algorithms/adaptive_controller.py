"""
This file holds the required logic for the adaptive controllers used in the thesis.
These are:
- Augmented Reward-Biased Maximum Likelihood Estimator
- Stabilizing Learning
- Proximal Policy Optimization

@author: Lucas Weitering
"""

import numpy as np
import control
import scipy.linalg
import scipy
import torch
import torch.distributions as dist

import matplotlib.pyplot as plt
import os


class AdaptiveController():
    """
        Implements adaptive controllers for linear quadratic control problems.

        This class supports the following algorithms:
        - Augmented Reward-Biased Maximum Likelihood Estimator (ARBMLE)
        - Stabilizing Learning (STABL)
        - Proximal Policy Optimization (PPO)

        It provides methods for parameter estimation, gain matrix computation, and running experiments.
        """
    def __init__(self, A, B, Q, R, horizon, T_init, reg, algorithm, cur_seed,
                 num_restarts=5, max_iters=500, step_size=0.05, rel_tol=1e-5,
                 delta=1e-3, bias=0.01, T_w=35, sigma_w=2, S=100, plot_actions=False, plot_costs=False,
                 plot_states=False, nonlinearity=False, nonlinear_case=None, problem_name_nonlin=None):
        """
            Initializes the AdaptiveController with system parameters and algorithm settings.

            Parameters:
            - A (ndarray): State transition matrix of the true system.
            - B (ndarray): Control input matrix of the true system.
            - Q (ndarray): State cost matrix.
            - R (ndarray): Control input cost matrix.
            - horizon (int): Total time steps for the experiment.
            - T_init (int): Initial time steps using a stabilizing controller.
            - reg (float): Regularization parameter for least squares estimation.
            - algorithm (str): Algorithm to use ('ARBMLE', 'STABL', or 'OPTIMAL').
            - cur_seed (int): Random seed for reproducibility.
            - num_restarts (int, optional): Number of restarts for gradient descent.
            - max_iters (int, optional): Maximum iterations for gradient descent.
            - step_size (float, optional): Step size for gradient descent.
            - rel_tol (float, optional): Relative tolerance for convergence.
            - delta (float, optional): Confidence level for confidence intervals.
            - bias (float, optional): Bias parameter for ARBMLE.
            - T_w (int, optional): Window size for STABL algorithm.
            - sigma_w (float, optional): Additive noise standard deviation for STABL.
            - S (int, optional): Upper bound on the noise norm.
            - plot_actions (bool, optional): Whether to plot control actions.
            - plot_costs (bool, optional): Whether to plot costs.
            - plot_states (bool, optional): Whether to plot states.
            - nonlinearity (bool, optional): Whether to include nonlinearity in the system.
            - nonlinear_case (str, optional): Type of nonlinearity.
            - problem_name_nonlin (str, optional): Name of the nonlinear problem.
            """

        # Set system dynamics
        np.random.seed(cur_seed)
        self.cur_seed = cur_seed
        self.A_star = A
        self.B_star = B
        self.theta_star = np.hstack((A, B))
        self.n, self.m = B.shape
        self.nonlinearity = nonlinearity
        self.nonlinear_case = nonlinear_case
        self.problem_name_nonlin = problem_name_nonlin

        # Control matrices and parameters
        self.Q = Q
        self.R = R
        self.horizon = horizon
        self.T_init = T_init
        self.reg = reg
        self.rl_algo = algorithm

        # Initialize gain matrices (also accounting for nonlinear linearithmic case) and setting nonlinear constant c
        self.constant_nonlin = None
        A_substitute = np.copy(self.A_star)
        # Handle nonlinearity cases and set constant_nonlin based on the problem
        if self.nonlinearity:
            if self.nonlinear_case == "x_sin(x)":
                if self.problem_name_nonlin == "Boeing":
                    self.constant_nonlin = 1.1
                elif self.problem_name_nonlin in ["UAV", "CartPole"]:
                    self.constant_nonlin = 0.1
            elif self.nonlinear_case == "x_squared":
                if self.problem_name_nonlin == "Boeing":
                    self.constant_nonlin = 0.002
                else:
                    self.constant_nonlin = 0.01
            elif nonlinear_case == "x_log(x)":
                if self.problem_name_nonlin == "CartPole":
                    self.constant_nonlin = 0.0005
                elif self.problem_name_nonlin == "Boeing":
                    self.constant_nonlin = 0.001
                elif self.problem_name_nonlin == "UAV":
                    self.constant_nonlin = 0.003

                # Modify the A_substitute matrix to account for the nonlinearity
                # Add c*log(2) to the diagonal of A_star
                c_log_2_term = np.log(2) * self.constant_nonlin
                np.fill_diagonal(A_substitute, np.diagonal(A_substitute) + c_log_2_term)
                print(f"Adapted LQR Controller at work! Nonlinearity: {nonlinear_case}")
                # print(self.A_star)
            else:
                print(f"Unknown Nonlinearity: {self.nonlinear_case}")
                raise Exception
        self.K_star, _, _ = control.dlqr(A_substitute, B, self.Q, self.R)
        self.K_init, _, _ = control.dlqr(self.A_star, self.B_star, 10 * np.eye(self.n), np.eye(self.m))  # Stabilizing controller

        # PGD parameters
        self.num_restarts = num_restarts
        self.max_iters = max_iters
        self.step_size = step_size
        self.rel_tol = rel_tol

        # Confidence Interval Parameters
        self.delta = delta
        self.S = S

        # RBMLE parameters
        self.bias = bias

        # STABL parameters
        self.T_w = T_w
        self.sigma_w = sigma_w

        # Initial settings
        self.states = np.zeros((self.n, self.horizon + self.T_init))
        self.inputs = np.zeros((self.m, self.horizon + self.T_init))
        self.costs = np.zeros((self.horizon + self.T_init,))
        self.cov = self.reg * np.eye(self.n + self.m)
        self.x = self.states[:, 0]
        self.u = self.inputs[:, 0]
        self.t = 0
        self.theta_emp = np.zeros_like(self.theta_star)
        self.K = np.zeros((self.K_star.shape[0], self.K_star.shape[1], self.horizon + self.T_init))
        self.theta = np.zeros((self.theta_star.shape[0], self.theta_star.shape[1], self.horizon + self.T_init))
        self.J = np.zeros(self.horizon + self.T_init)
        self.event = np.ones(self.horizon + self.T_init)

        # Monitoring
        self.plot_actions = plot_actions
        self.plot_states = plot_states
        self.plot_costs = plot_costs
        self.state_costs_recording = None
        self.input_costs_recording = None
        self.timesteps_recording = None

    def project_weighted_ball(self, M, theta_center, cov, eps):
        """
        Projects matrix M onto a weighted Frobenius norm ball centered at theta_center.

        Parameters:
        - M (ndarray): The matrix to be projected.
        - theta_center (ndarray): Center of the projection ball.
        - cov (ndarray): Covariance matrix used for weighting.
        - eps (float): Squared radius of the ball.

        Returns:
        - theta_star (ndarray): The projected matrix.
        """

        # If M is already centered, adjust the projection accordingly
        assert len(M.shape) == 2
        assert M.shape == theta_center.shape
        assert len(cov.shape) == 2
        assert cov.shape[0] == cov.shape[1]
        assert M.shape[1] == cov.shape[0]
        assert self.eps > 0

        if not np.allclose(theta_center, np.zeros_like(theta_center)):
            ret = self.project_weighted_ball(M - theta_center, np.zeros_like(theta_center), cov, eps)
            ret += theta_center
            return ret

        # Check if M is within the ball; if so, return M
        if np.trace(M.dot(cov).dot(M.T)) <= eps:
            return M

        # Compute the projection using eigen decomposition
        # The solution takes the form theta_star = M (I + lambda * cov)^{-1}
        w, V = np.linalg.eigh(cov)

        # Find lambda such that the constraint is met
        # Tr( M * (I + lambda * cov)^{-1} cov * (I + lambda * cov)^{-1} M.T ) = eps
        MV = M.dot(V)
        VTMT_MV = MV.T.dot(MV)
        term2 = np.diag(VTMT_MV)

        def func(lam):
            # Function to find the root for lambda
            assert lam >= 0
            term1 = (w / ((1 + lam * w) ** 2))
            val = eps - np.sum(term1 * term2)
            if np.abs(val) < 1e-5:
                val = 0
            return val

        # Initialize lambda bounds and solve for lambda
        lam_ub = 1
        lam_lb = 0
        while func(lam_lb) <= 0 and func(2 * lam_ub) < 0:
            lam_lb = lam_ub
            lam_ub *= 2

        lam_star, _ = scipy.optimize.brentq(func, lam_lb, lam_ub, full_output=True)

        # Compute the projected matrix
        theta_star = MV.dot(np.diag(1 / (1 + lam_star * w))).dot(V.T)

        return theta_star

    def pd_inv_sqrt(self, P):
        """
        Compute the inverse square root of a positive definite matrix P.

        Parameters:
            P (np.ndarray): Symmetric positive definite matrix.

        Returns:
            np.ndarray: Inverse square root of P.
        """
        assert len(P.shape) == 2
        assert P.shape[0] == P.shape[1]
        w, v = np.linalg.eigh(P)
        TOL = 1e-5
        w = np.maximum(w, TOL)
        return v.dot(np.diag(1 / np.sqrt(w))).dot(v.T)

    def get_gain_matrix(self, theta):
        """
        Compute the LQR gain matrix K, solution P to the Riccati equation, and cost J for given system parameters.

        Parameters:
            theta (np.ndarray): Combined system parameters [A | B], where A and B are concatenated horizontally.

        Returns:
            K (np.ndarray): LQR gain matrix.
            P (np.ndarray): Solution to the discrete-time Riccati equation.
            J (float): Trace of P, representing the estimated cost.

        """
        A = theta[:, :self.n]
        B = theta[:, self.n:]
        K, P, _ = control.dlqr(A, B, self.Q, self.R)
        J = np.trace(P)
        return K, P, J

    def get_input(self, K):
        return -1 * np.dot(K, self.x)

    def get_next_state(self, w):
        """
        Compute the next state of the system, applying optional nonlinearities.

        Parameters:
            w (np.ndarray): Process noise vector.

        Returns:
            np.ndarray: The next state vector.
        """
        next_state = np.dot(self.A_star, self.x) + np.dot(self.B_star, self.u) + w
        # Handle nonlinearity cases
        if self.nonlinearity:
            # Set constant_nonlin based on the problem
            c = self.constant_nonlin
            if self.nonlinear_case == "x_sin(x)":
                next_state += c * self.x * np.sin(self.x)
            elif self.nonlinear_case == "x_squared":
                next_state += c * self.x * self.x
            elif self.nonlinear_case == "x_log(x)":
                # x log(sgn(x) * (x + 2 * H(x)))
                h_of_x = [-1 if self.x[i] < 0 else 1 for i in range(len(self.x))]
                inner_term = self.x + np.multiply(h_of_x, 2)
                next_state += c * self.x * np.log(h_of_x * inner_term)
        return next_state

    def calculate_cost(self, x, u):
        return np.dot(np.dot(x.T, self.Q), x) + np.dot(np.dot(u.T, self.R), u)

    def get_lse(self, reg):
        """ Returns the least-squares estimate for the model-based algorithms using the regularization parameter."""
        X1 = np.dot(self.Z.T, self.Z) + reg * np.eye(self.n + self.m)
        X2 = np.dot(self.Z.T, self.X)
        theta_hat, _, _, _ = np.linalg.lstsq(X1, X2, rcond=None)
        return theta_hat.T

    def get_estimate(self, reg):
        self.theta_emp = self.get_lse(reg)
        if self.rl_algo == 'ARBMLE':
            return self.get_aug_rbmle(reg)
        elif self.rl_algo == 'STABL':
            return self.get_stbl(reg)
        elif self.rl_algo == 'OPTIMAL':
            return self.theta_star

    def confidence_interval(self):
        """
       Calculates the confidence interval (epsilon) used in parameter estimation.

       Uses concentration inequalities based on the determinant of the covariance matrix.

       Returns:
       - eps (float): The computed confidence interval.
       """
        term1 = np.sqrt(np.linalg.det(self.cov))
        term2 = np.sqrt(np.linalg.det(self.reg * np.eye(self.n + self.m)))
        term3 = term1 / (term2 * self.delta)
        term4 = self.n * np.sqrt(2 * np.log(term3))
        term5 = np.sqrt(self.reg) * self.S
        return (term4 + term5) ** 2

    def update_history(self):
        self.states[:, self.t] = self.x
        self.inputs[:, self.t - 1] = self.u
        self.costs[self.t - 1] = self.c
        z = np.concatenate((self.states[:, self.t - 1], self.u))
        self.cov += np.outer(z, z)
        self.eps = self.confidence_interval()
        self.X = (self.states[:, 1:self.t]).T
        self.Z = np.concatenate((self.states[:, :self.t - 1], self.inputs[:, :self.t - 1])).T

    def run_experiment(self, noise, input_noise, reg):
        """
        Runs the adaptive control experiment using the specified algorithm.

        Parameters:
        - noise (ndarray): Process noise for each time step.
        - input_noise (ndarray): Input noise for exploration during the initial phase.
        - reg (float): Regularization parameter for parameter estimation.
        """
        all_actions, all_states = [], []
        all_states.append(self.x)
        all_actions.append(self.u)
        try:
            # Initial exploration phase using a stabilizing controller
            while self.t < self.T_init:
                # Apply stabilizing control input with added exploration noise
                self.u = self.get_input(self.K_init) + input_noise[:, self.t]
                all_actions.append(self.u)
                # Calculate cost and update state
                self.c = self.calculate_cost(self.x, self.u)
                self.x = self.get_next_state(noise[:, self.t])
                all_states.append(self.x)
                self.t += 1
                self.update_history()

            # Main adaptive control loop
            while self.t < self.horizon + self.T_init - 1:
                # Estimate system parameters and compute control gain
                theta_t = self.get_estimate(reg)
                Kt, _, J = self.get_gain_matrix(theta_t)
                last_cov = np.linalg.det(self.cov)
                episode_length = 0

                # Continue until sufficient new information is gathered
                while ((np.linalg.det(self.cov) <= 2 * last_cov) or (episode_length < 10)) and self.t < self.horizon + self.T_init - 1:
                    # Apply control input using the current gain matrix
                    self.u = self.get_input(Kt)
                    all_actions.append(self.u)
                    # Add exploration noise for STABL algorithm during the initial window
                    if self.rl_algo == 'STABL' and self.t <= self.T_init + self.T_w:
                        self.u += np.random.normal(0, self.sigma_w)
                    # Calculate cost and update state
                    self.c = self.calculate_cost(self.x, self.u)
                    self.x = self.get_next_state(noise[:, self.t])
                    all_states.append(self.x)
                    self.t += 1
                    episode_length += 1
                    self.update_history()
        except Exception as e:
            print(e)
            if self.nonlinearity:
                return
            else:
                raise e

        if self.plot_costs:
            states_np = np.array(all_states)
            actions_np = np.array(all_actions)

            state_costs = []
            input_costs = []
            total_costs = []

            for t in range(states_np.shape[0]):
                state_cost = states_np[t, :].T @ self.Q @ states_np[t, :]  # state * Q * state
                input_cost = actions_np[t, :].T @ self.R @ actions_np[t, :]  # input * R * input
                total_cost = state_cost + input_cost  # cost_t

                state_costs.append(state_cost)
                input_costs.append(input_cost)
                total_costs.append(total_cost)

            timesteps = range(states_np.shape[0])

            self.state_costs_recording = state_costs
            self.input_costs_recording = input_costs
            self.timesteps_recording = timesteps

            # 1. Create stacked subplots
            fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

            # Plot state cost
            ax[0].plot(timesteps, state_costs, label="State Cost $(x * Q * x)$", color='b')
            ax[0].set_title(f"State Cost for {self.problem_name_nonlin} {self.rl_algo}")
            ax[0].set_ylabel("Cost")
            ax[0].legend()

            # Plot input cost
            ax[1].plot(timesteps, input_costs, label="Input Cost $(u * R * u)$", color='g')
            ax[1].set_title(f"Input Cost for {self.problem_name_nonlin} {self.rl_algo}")
            ax[1].set_ylabel("Cost")
            ax[1].legend()

            # Plot total cost
            ax[2].plot(timesteps, total_costs, label="Total Cost $c(x,u)$", color='r', linestyle='--')
            ax[2].set_title(f"Total Cost for {self.problem_name_nonlin} {self.rl_algo}")
            ax[2].set_xlabel("Timestep")
            ax[2].set_ylabel("Cost")
            ax[2].legend()

            plt.tight_layout()

            # Create directory if it doesn't exist
            output_dir = os.path.join("plots", "cost_analysis")
            os.makedirs(output_dir, exist_ok=True)

            # Save the stacked version
            filename_stacked = f"cost_breakdown_stacked_{self.problem_name_nonlin}_{self.rl_algo}_1.pdf"
            file_path_stacked = os.path.join(output_dir, filename_stacked)
            plt.savefig(file_path_stacked)
            plt.close()

            print(f"Stacked cost breakdown plot saved at: {file_path_stacked}")

            # 2. Create combined plot with all curves on the same graph
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps, state_costs, label="State Cost $(x * Q * x)$", color='b')
            plt.plot(timesteps, input_costs, label="Input Cost $(u * R * u)$", color='g')
            plt.plot(timesteps, total_costs, label="Total Cost $c(x,u)$", color='r', linestyle='--')

            plt.title(f"Cost Breakdown {self.problem_name_nonlin} {self.rl_algo}")
            plt.xlabel("Timestep")
            plt.ylabel("Cost")
            plt.legend()
            plt.tight_layout()

            # Save the combined version
            filename_combined = f"cost_breakdown_{self.problem_name_nonlin}_{self.rl_algo}_1.pdf"
            file_path_combined = os.path.join(output_dir, filename_combined)
            plt.savefig(file_path_combined)
            plt.close()

            print(f"Combined cost breakdown plot saved at: {file_path_combined}")

            sum_alg_state = np.cumsum(state_costs[199:])
            sum_alg_input = np.cumsum(input_costs[199:])
            sum_alg_total = np.cumsum(total_costs[199:])

            print(
                f"For alg: {self.rl_algo} and problem: {self.problem_name_nonlin}\nstate sum: {sum_alg_state[-1]}"
                f"\ninput_sum: {sum_alg_input[-1]}\ntotal sum: {sum_alg_total[-1]}")

        if self.plot_actions:
            # Plot the actions
            actions_np = torch.tensor(np.array(all_actions),
                                      dtype=torch.float32).detach().cpu().numpy()  # Convert tensor to numpy array
            timesteps = actions_np.shape[0]
            action_dim = actions_np.shape[1]

            time = range(timesteps)

            # Plot each component of the action vector
            fig, ax = plt.subplots(action_dim, 1, figsize=(10, 8))

            for i in range(action_dim):
                if isinstance(ax, np.ndarray):
                    ax[i].plot(time, actions_np[:, i])
                    ax[i].set_title(f"Action Component {i + 1} Over Time")
                    ax[i].set_xlabel("Timestep")
                    ax[i].set_ylabel(f"Action Component {i + 1}")
                else:  # In case there exists only one input dimension
                    ax.plot(time, actions_np[:, i])
                    ax.set_title(f"Action Component {i + 1} Over Time")
                    ax.set_xlabel("Timestep")
                    ax.set_ylabel(f"Action Component {i + 1}")

            plt.tight_layout()
            plt.show()

        if self.plot_states:
            # Plot the states
            states_np = torch.tensor(np.array(all_states), dtype=torch.float32).detach().cpu().numpy()
            timesteps = states_np.shape[0]
            state_dim = states_np.shape[1]
            time = range(timesteps)

            # Plot each component of the state vector
            fig, ax = plt.subplots(state_dim, 1, figsize=(10, 8))

            for i in range(state_dim):
                if isinstance(ax, np.ndarray):
                    ax[i].plot(time, states_np[:, i])
                    ax[i].set_title(f"State Component {i + 1} Over Time")
                    ax[i].set_xlabel("Timestep")
                    ax[i].set_ylabel(f"State Component {i + 1}")
                else:  # In case there exists only one state dimension
                    ax.plot(time, states_np[:, i])
                    ax.set_title(f"State Component {i + 1} Over Time")
                    ax.set_xlabel("Timestep")
                    ax.set_ylabel(f"State Component {i + 1}")
            plt.tight_layout()
            plt.show()

    def squared_error(self, theta):
        """
       Compute the regularized sum of squared errors between predicted and actual states.

       Parameters:
           theta (np.ndarray): Parameter matrix used for state predictions.

       Returns:
           float: The total error, including the regularization term.
       """
        temp1 = self.states[:, 1:self.t] - np.dot(theta, self.Z.T)
        error = np.sum(np.linalg.norm(temp1, axis=0) ** 2)
        error += self.reg * np.linalg.norm(theta)
        return error

    def lse_gradient(self, theta):
        grad = 2 * (np.dot(theta, (np.dot(self.Z.T, self.Z) + self.reg * np.eye(self.n + self.m)).T) - np.dot(self.X.T,
                                                                                                              self.Z))
        return grad

    def get_stbl(self, reg):
        self.function = self.cost
        self.gradient = self.cost_gradient
        theta = self.pgd(self.eps, reg)
        return theta

    def get_aug_rbmle(self, reg):
        self.function = self.rbmle
        self.gradient = self.rbmle_gradient
        theta = self.pgd(self.eps, reg)
        return theta

    def rbmle(self, theta):
        rbmle_cost = self.bias * (self.horizon ** 0.5) * self.cost(theta) + self.squared_error(theta)
        return rbmle_cost

    def rbmle_gradient(self, theta):
        rbmle_grad = self.bias * (self.horizon ** 0.5) * self.cost_gradient(theta) + self.lse_gradient(theta)
        return rbmle_grad

    def cost(self, theta):
        _, _, cost = self.get_gain_matrix(theta)
        return cost

    def cost_gradient(self, theta):
        """
        Compute the gradient of the cost function with respect to system parameters theta.

        Parameters:
            theta (np.ndarray): Combined system parameters [A | B].

        Returns:
            np.ndarray: Gradient matrix of the cost function.
        """
        A = theta[:, :self.n]
        B = theta[:, self.n:]
        K, P, J = self.get_gain_matrix(theta)
        A_c = A + B.dot(K)
        grad = np.zeros((self.n, self.n + self.m))

        for idx in range(self.n):
            for jdx in range(self.n + self.m):
                U = np.zeros((self.n, self.n + self.m))
                U[idx, jdx] = 1
                target = A_c.T.dot(P.dot(U)).dot(np.vstack((np.eye(self.n), K)))
                target += target.T
                DU = scipy.linalg.solve_discrete_lyapunov(A.T, self.Q, None)
                grad[idx, jdx] = np.trace(DU)

        return grad

    def pgd(self, eps, reg):
        """
        Perform projected gradient descent to minimize the objective function.

        Parameters:
            cov (np.ndarray): Covariance matrix for the weighted projection.
            eps (float): Radius of the weighted ball for projection.
            reg (float): Regularization parameter.

        Returns:
            np.ndarray: Optimal parameter vector obtained after PGD.
        """
        theta_initial = self.get_lse(reg)
        theta_opt = theta_initial
        f_opt = self.function(theta_opt)

        for i in range(self.num_restarts):
            theta_cur = theta_initial + np.sign(i) * np.random.normal(0, 1, size=theta_initial.shape)
            theta_cur = self.project_weighted_ball(theta_cur, self.theta_emp, self.cov, eps)
            f_cur = self.function(theta_cur)

            for _ in range(self.max_iters):
                grad = self.gradient(theta_cur)
                theta_next = theta_cur - self.step_size * grad / np.linalg.norm(grad)
                theta_next = self.project_weighted_ball(theta_next, self.theta_emp, self.cov, eps)
                f_next = self.function(theta_next)

                if f_next <= f_opt:
                    f_opt = f_next
                    theta_opt = theta_next

                if (f_cur - f_next) / f_cur < self.rel_tol:
                    break

                theta_cur = theta_next
                f_cur = f_next

        return theta_opt

    def run_optimal(self, noise, input_noise):
        """ Similar method to run_experiment. Running a given experiment using the optimal controller. """
        all_actions, all_states = [], []
        all_states.append(self.x)
        all_actions.append(self.u)
        # Initial phase: use stabilizing controller (self.K_init)
        try:
            while self.t < self.T_init:
                self.u = self.get_input(self.K_init) + input_noise[:, self.t]  # Apply the stabilizing controller
                all_actions.append(self.u)
                self.c = self.calculate_cost(self.x, self.u)  # Calculate cost
                self.x = self.get_next_state(noise[:, self.t])  # Update state with noise
                all_states.append(self.x)
                self.t += 1
                self.update_history()  # Update history with current data

            # Main phase: use optimal feedback gain matrix (self.K_star)
            while self.t < self.horizon + self.T_init - 1:
                self.u = self.get_input(self.K_star)  # Apply the optimal feedback controller
                all_actions.append(self.u)
                self.c = self.calculate_cost(self.x, self.u)  # Calculate cost
                self.x = self.get_next_state(noise[:, self.t])  # Update state with noise
                all_states.append(self.x)
                self.t += 1
                self.update_history()  # Update history with current data
        except Exception as e:
            print(e)
            if self.nonlinearity:
                return
            else:
                raise e

        if self.plot_costs:
            # Convert recorded states and actions into numpy arrays for easier manipulation
            states_np = np.array(all_states)
            actions_np = np.array(all_actions)

            # Initialize lists to store the cost components
            state_costs = []
            input_costs = []
            total_costs = []

            # Loop through each timestep and compute the two parts of the cost
            for t in range(states_np.shape[0]):
                state_cost = states_np[t, :].T @ self.Q @ states_np[t, :]  # state * Q * state
                input_cost = actions_np[t, :].T @ self.R @ actions_np[t, :]  # input * R * input
                total_cost = state_cost + input_cost

                state_costs.append(state_cost)
                input_costs.append(input_cost)
                total_costs.append(total_cost)

            timesteps = range(states_np.shape[0])

            self.state_costs_recording = state_costs
            self.input_costs_recording = input_costs
            self.timesteps_recording = timesteps

            # 1. Create stacked subplots
            fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

            # Plot state cost
            ax[0].plot(timesteps, state_costs, label="State Cost $(x * Q * x)$", color='b')
            ax[0].set_title(f"State Cost for {self.problem_name_nonlin} {self.rl_algo}")
            ax[0].set_ylabel("Cost")
            ax[0].legend()

            # Plot input cost
            ax[1].plot(timesteps, input_costs, label="Input Cost $(u * R * u)$", color='g')
            ax[1].set_title(f"Input Cost for {self.problem_name_nonlin} {self.rl_algo}")
            ax[1].set_ylabel("Cost")
            ax[1].legend()

            # Plot total cost
            ax[2].plot(timesteps, total_costs, label="Total Cost $c(x,u)$", color='r', linestyle='--')
            ax[2].set_title(f"Total Cost for {self.problem_name_nonlin} {self.rl_algo}")
            ax[2].set_xlabel("Timestep")
            ax[2].set_ylabel("Cost")
            ax[2].legend()

            plt.tight_layout()

            # Create directory if it doesn't exist
            output_dir = os.path.join("plots", "cost_analysis")
            os.makedirs(output_dir, exist_ok=True)

            # Save the stacked version
            filename_stacked = f"cost_breakdown_stacked_{self.problem_name_nonlin}_{self.rl_algo}_1.pdf"
            file_path_stacked = os.path.join(output_dir, filename_stacked)
            plt.savefig(file_path_stacked)
            plt.close()

            print(f"Stacked cost breakdown plot saved at: {file_path_stacked}")

            # 2. Create combined plot with all curves on the same graph
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps, state_costs, label="State Cost $(x * Q * x)$", color='b')
            plt.plot(timesteps, input_costs, label="Input Cost $(u * R * u)$", color='g')
            plt.plot(timesteps, total_costs, label="Total Cost $c(x,u)$", color='r', linestyle='--')

            plt.title(f"Cost Breakdown {self.problem_name_nonlin} {self.rl_algo}")
            plt.xlabel("Timestep")
            plt.ylabel("Cost")
            plt.legend()
            plt.tight_layout()

            # Save the combined version
            filename_combined = f"cost_breakdown_{self.problem_name_nonlin}_{self.rl_algo}_1.pdf"
            file_path_combined = os.path.join(output_dir, filename_combined)
            plt.savefig(file_path_combined)
            plt.close()

            print(f"Combined cost breakdown plot saved at: {file_path_combined}")

            sum_alg_state = np.cumsum(state_costs[199:])
            sum_alg_input = np.cumsum(input_costs[199:])
            sum_alg_total = np.cumsum(total_costs[199:])

            print(f"For alg: {self.rl_algo} and problem: {self.problem_name_nonlin}\nstate sum: {sum_alg_state[-1]}"
                  f"\ninput_sum: {sum_alg_input[-1]}\ntotal sum: {sum_alg_total[-1]}")

        # Optionally plot actions and states
        if self.plot_actions:
            actions_np = torch.tensor(np.array(all_actions),
                                      dtype=torch.float32).detach().cpu().numpy()  # Convert tensor to numpy array
            timesteps = actions_np.shape[0]
            action_dim = actions_np.shape[1]
            time = range(timesteps)
            fig, ax = plt.subplots(action_dim, 1, figsize=(10, 8))
            for i in range(action_dim):
                if isinstance(ax, np.ndarray):
                    ax[i].plot(time, actions_np[:, i])
                    ax[i].set_title(f"Action Component {i + 1} Over Time")
                    ax[i].set_xlabel("Timestep")
                    ax[i].set_ylabel(f"Action Component {i + 1}")
                else:
                    ax.plot(time, actions_np[:, i])
                    ax.set_title(f"Action Component {i + 1} Over Time")
                    ax.set_xlabel("Timestep")
                    ax.set_ylabel(f"Action Component {i + 1}")
            plt.tight_layout()
            plt.show()

        if self.plot_states:
            states_np = torch.tensor(np.array(all_states), dtype=torch.float32).detach().cpu().numpy()
            timesteps = states_np.shape[0]
            state_dim = states_np.shape[1]
            time = range(timesteps)
            fig, ax = plt.subplots(state_dim, 1, figsize=(10, 8))
            for i in range(state_dim):
                if isinstance(ax, np.ndarray):
                    ax[i].plot(time, states_np[:, i])
                    ax[i].set_title(f"State Component {i + 1} Over Time")
                    ax[i].set_xlabel("Timestep")
                    ax[i].set_ylabel(f"State Component {i + 1}")
                else:
                    ax.plot(time, states_np[:, i])
                    ax.set_title(f"State Component {i + 1} Over Time")
                    ax.set_xlabel("Timestep")
                    ax.set_ylabel(f"State Component {i + 1}")
            plt.tight_layout()
            plt.show()


# A number of different classes and methods required for PPO
class LinearPolicy(torch.nn.Module):
    """
    Linear policy network for PPO, mapping states to action distributions.
    """
    def __init__(self, state_dim, action_dim, initial_log_std=-4, std_clamp_max=0.5):
        super(LinearPolicy, self).__init__()
        # Linear mapping from state to mean of the action
        self.linear = torch.nn.Linear(state_dim, action_dim, bias=False)

        # Log standard deviation, initialized to a very low value (close to zero)
        # This will create a narrow distribution with small variance.
        self.log_std = torch.nn.Parameter(torch.ones(action_dim) * initial_log_std)
        self.counter = 0
        self.std_clamp_max = std_clamp_max

    def forward(self, state):
        # Mean of the action form the linear layer
        mean = -self.linear(state)

        # Convert log standard deviation to standard deviation
        std = torch.exp(self.log_std)

        # Ensure that standard deviation is small enough for narrow distribution
        std = torch.clamp(std, min=1e-6, max=self.std_clamp_max)  # Ensures std does not get too large
        self.counter += 1

        # Check for NaN values (debugging)
        if torch.isnan(mean).any() or torch.isnan(std).any():
            print("Warning: mean or std contains NaN values. (after {} times through forward)".format(self.counter))
            print(f"mean: {mean}, std: {std}")
            print("state: {}".format(state))

        # Return a Normal (Gaussian) distribution with mean and std
        return dist.Normal(mean, std)


class ValueNetwork(torch.nn.Module):
    """
    Value network for PPO, approximating the state-value function V(s).
    """
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        # P is a learnable matrix (symmetric positive definite)
        self.P = torch.nn.Parameter(torch.eye(state_dim, dtype=torch.float32))

    def forward(self, state):
        # Ensuring that `state` has a batch dimension
        # Calculate V(x) = x^T P x for a batch of states
        value = torch.einsum('bi,ij,bj->b', state.to(torch.float32), self.P, state.to(torch.float32))
        value = value.unsqueeze(-1)  # Return as a column vector (shape `[batch_size, 1]`)
        return value


class ProximalPolicyOptimization(AdaptiveController):
    """
    Extends AdaptiveController to implement the Proximal Policy Optimization (PPO) algorithm.

    Parameters:
    - gamma (float, optional): Discount factor for rewards.
    - lambda_ (float, optional): GAE (Generalized Advantage Estimation) parameter.
    - epsilon (float, optional): Clipping parameter for PPO.
    - lr_policy (float, optional): Learning rate for the policy network.
    - lr_value (float, optional): Learning rate for the value network.
    - std_clamp_max (float, optional): Maximum standard deviation for action distribution.
    - update_steps (int, optional): Number of steps between policy updates.
    - Additional parameters inherited from AdaptiveController.
    """
    def __init__(self, A, B, Q, R, horizon, T_init, reg, algorithm, cur_seed,
                 gamma=1.0, lambda_=1.0, epsilon=0.2, lr_policy=1e-3, lr_value=1e-3, std_clamp_max=0.5,
                 update_steps=10, plot_actions=False, plot_costs=False, plot_states=False, nonlinearity=False,
                 nonlinear_case=None, **kwargs):
        # Initialize AdaptiveController with additional parameters from kwargs
        super().__init__(A, B, Q, R, horizon, T_init, reg, algorithm, cur_seed, plot_actions=plot_actions,
                         plot_costs=plot_costs, plot_states=plot_states, nonlinearity=nonlinearity,
                         nonlinear_case=nonlinear_case, **kwargs)

        # PPO specific parameters
        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.update_steps = update_steps

        # Initialize the policy and value networks
        self.policy = LinearPolicy(self.n, self.m, std_clamp_max=std_clamp_max)
        self.value_net = ValueNetwork(self.n)

        # Define optimizers for both networks
        self.optimizer_policy = torch.optim.Adam(self.policy.parameters(), lr=lr_policy)
        self.optimizer_value = torch.optim.Adam(self.value_net.parameters(), lr=lr_value)

    def ppo_loss(self, old_policy, states, actions, advantages):
        """
        Computes the PPO loss function.

        Parameters:
        - old_policy (LinearPolicy): The policy from the previous update.
        - states (Tensor): Batch of states.
        - actions (Tensor): Batch of actions taken.
        - advantages (Tensor): Advantage estimates for each state-action pair.

        Returns:
        - loss (Tensor): The computed PPO loss.
        """
        # Get the action distribution from the current policy
        new_dist = self.policy(states)

        # Calculate the log probabilities of the actions
        new_log_probs = new_dist.log_prob(actions).sum(dim=-1)

        if old_policy is None:
            # If no old policy, skip policy ratio and directly use new log probabilities
            loss = -advantages * new_log_probs
        else:
            # Get the action distribution from the old policy
            old_dist = old_policy(states)
            old_log_probs = old_dist.log_prob(actions).sum(dim=-1).detach()

            # Compute the ratio between new and old probabilities
            ratios = torch.exp(new_log_probs - old_log_probs)

            # Compute the clipped surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
            loss = -torch.min(surr1, surr2).mean()

        return loss.mean()

    def initialise_policy_least_squares(self, states, actions):
        # states: (T, n) tensor, where T is the number of time steps, n is the state dimension
        # actions: (T, m) tensor, where T is the number of time steps, m is the action dimension

        # Perform least squares estimation: K = (X'X)^-1 X'U, where X are the states and U are the actions
        X = states.cpu().numpy()
        U = actions.cpu().numpy()

        # Use least-squares to solve for K: U = -KX
        K = np.linalg.lstsq(X, U, rcond=None)[0].T  # Transpose to get correct dimensions

        # Initialize the policy network with the estimated K
        self.policy.linear.weight.data = torch.tensor(-K, dtype=torch.float32)

    def compute_advantages(self, rewards, values):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.lambda_ * gae
            advantages.insert(0, gae)
        return torch.tensor(advantages)

    def update_policy(self, old_policy, states, actions, returns, advantages):
        # Update policy (actor)
        self.optimizer_policy.zero_grad()
        policy_loss = self.ppo_loss(old_policy, states, actions, advantages)
        policy_loss.backward()
        self.optimizer_policy.step()

        # Update value function (critic)
        self.optimizer_value.zero_grad()
        value_loss = torch.nn.MSELoss()(self.value_net(states).squeeze(), returns)
        value_loss.backward()
        self.optimizer_value.step()

    def ppo_update(self, old_policy, states, actions, rewards, next_states):
        # Compute value estimates for states and next states
        values = self.value_net(states).squeeze()
        next_values = self.value_net(next_states).squeeze()

        # Compute returns (discounted rewards)
        returns = rewards + self.gamma * next_values

        # Compute advantages using GAE
        advantages = self.compute_advantages(rewards, values)

        # Update both the policy (actor) and value network (critic)
        self.update_policy(old_policy, states, actions, returns, advantages)

    def run_experiment(self, noise, input_noise, reg):
        # Initial phase with stabilizing controller
        all_states, all_actions, rewards, next_states = [], [], [], []
        try:
            while self.t < self.T_init:
                self.u = self.get_input(self.K_init) + input_noise[:, self.t]
                self.c = self.calculate_cost(self.x, self.u)
                next_state = self.get_next_state(noise[:, self.t])

                all_states.append(self.x)
                all_actions.append(self.u)
                rewards.append(-self.c)
                next_states.append(next_state)

                self.x = next_state
                self.t += 1
                self.update_history()

            states = torch.tensor(np.array(all_states), dtype=torch.float32)
            actions = torch.tensor(all_actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)

            # Parametrise PPO algorithm
            # First use least-squares for initial parametrisation
            self.initialise_policy_least_squares(states, actions)

            old_policy = None
            self.ppo_update(old_policy, states, actions, rewards, next_states)

            # Learning phase: run and train PPO
            while self.t < self.horizon + self.T_init - 1:
                old_policy = LinearPolicy(self.n, self.m)
                old_policy.load_state_dict(self.policy.state_dict())

                states, actions, rewards, next_states = [], [], [], []

                for _ in range(self.update_steps):  # Define the number of steps before a policy update is done.
                    # Get the action distribution from the policy
                    distribution = self.policy(torch.tensor(self.x, dtype=torch.float32))

                    # Sample an action from the distribution
                    action = distribution.sample().detach().cpu().numpy()
                    self.u = action

                    # Apply the action and compute cost
                    self.c = self.calculate_cost(self.x, self.u)
                    reward = -self.c
                    next_state = self.get_next_state(noise[:, self.t])

                    # Store experience
                    states.append(self.x)
                    all_states.append(self.x)
                    actions.append(action)
                    all_actions.append(action)
                    rewards.append(reward)
                    next_states.append(next_state)

                    # Move to next state
                    self.x = next_state

                    # Make step in time
                    self.t += 1
                    if self.t >= self.horizon + self.T_init:
                        break

                    # Update monitoring
                    self.update_history()

                # Convert lists to tensors
                states = torch.tensor(states, dtype=torch.float32)
                actions_tensor = torch.tensor(actions, dtype=torch.float32)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)

                # Perform PPO update
                self.ppo_update(old_policy, states, actions_tensor, rewards, next_states)
        except Exception as e:
            print(e)
            if self.nonlinearity:
                return
            else:
                raise e

        if self.plot_costs:
            states_np = np.array(all_states)
            actions_np = np.array(all_actions)

            state_costs = []
            input_costs = []
            total_costs = []

            # Loop through each timestep and compute the two parts of the cost
            for t in range(states_np.shape[0]):
                state_cost = states_np[t, :].T @ self.Q @ states_np[t, :]  # state * Q * state
                input_cost = actions_np[t, :].T @ self.R @ actions_np[t, :]  # input * R * input
                total_cost = state_cost + input_cost

                state_costs.append(state_cost)
                input_costs.append(input_cost)
                total_costs.append(total_cost)

            timesteps = range(states_np.shape[0])

            self.state_costs_recording = state_costs
            self.input_costs_recording = input_costs
            self.timesteps_recording = timesteps

            # 1. Create stacked subplots
            fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

            # Plot state cost
            ax[0].plot(timesteps, state_costs, label="State Cost $(x * Q * x)$", color='b')
            ax[0].set_title(f"State Cost for {self.problem_name_nonlin} {self.rl_algo}")
            ax[0].set_ylabel("Cost")
            ax[0].legend()

            # Plot input cost
            ax[1].plot(timesteps, input_costs, label="Input Cost $(u * R * u)$", color='g')
            ax[1].set_title(f"Input Cost for {self.problem_name_nonlin} {self.rl_algo}")
            ax[1].set_ylabel("Cost")
            ax[1].legend()

            # Plot total cost
            ax[2].plot(timesteps, total_costs, label="Total Cost $c(x,u)$", color='r', linestyle='--')
            ax[2].set_title(f"Total Cost for {self.problem_name_nonlin} {self.rl_algo}")
            ax[2].set_xlabel("Timestep")
            ax[2].set_ylabel("Cost")
            ax[2].legend()

            plt.tight_layout()

            # Create directory if it doesn't exist
            output_dir = os.path.join("plots", "cost_analysis")
            os.makedirs(output_dir, exist_ok=True)

            # Save the stacked version
            filename_stacked = f"cost_breakdown_stacked_{self.problem_name_nonlin}_{self.rl_algo}_1.pdf"
            file_path_stacked = os.path.join(output_dir, filename_stacked)
            plt.savefig(file_path_stacked)
            plt.close()

            print(f"Stacked cost breakdown plot saved at: {file_path_stacked}")

            # 2. Create combined plot with all curves on the same graph
            plt.figure(figsize=(10, 6))
            plt.plot(timesteps, state_costs, label="State Cost $(x * Q * x)$", color='b')
            plt.plot(timesteps, input_costs, label="Input Cost $(u * R * u)$", color='g')
            plt.plot(timesteps, total_costs, label="Total Cost $c(x,u)$", color='r', linestyle='--')

            plt.title(f"Cost Breakdown {self.problem_name_nonlin} {self.rl_algo}")
            plt.xlabel("Timestep")
            plt.ylabel("Cost")
            plt.legend()
            plt.tight_layout()

            # Save the combined version
            filename_combined = f"cost_breakdown_{self.problem_name_nonlin}_{self.rl_algo}_1.pdf"
            file_path_combined = os.path.join(output_dir, filename_combined)
            plt.savefig(file_path_combined)
            plt.close()

            print(f"Combined cost breakdown plot saved at: {file_path_combined}")

            sum_alg_state = np.cumsum(state_costs[199:])
            sum_alg_input = np.cumsum(input_costs[199:])
            sum_alg_total = np.cumsum(total_costs[199:])

            print(
                f"For alg: {self.rl_algo} and problem: {self.problem_name_nonlin}\nstate sum: {sum_alg_state[-1]}"
                f"\ninput_sum: {sum_alg_input[-1]}\ntotal sum: {sum_alg_total[-1]}")

        if self.plot_actions:
            # Plot the actions
            actions_np = torch.tensor(np.array(all_actions), dtype=torch.float32).detach().cpu().numpy()  # Convert tensor to numpy array
            timesteps = actions_np.shape[0]
            action_dim = actions_np.shape[1]
            time = range(timesteps)

            # Plot each component of the action vector
            fig, ax = plt.subplots(action_dim, 1, figsize=(10, 8))

            for i in range(action_dim):
                if isinstance(ax, np.ndarray):
                    ax[i].plot(time, actions_np[:, i])
                    ax[i].set_title(f"Action Component {i + 1} Over Time")
                    ax[i].set_xlabel("Timestep")
                    ax[i].set_ylabel(f"Action Component {i + 1}")
                else:  # In case there exists only one input dimension
                    ax.plot(time, actions_np[:, i])
                    ax.set_title(f"Action Component {i + 1} Over Time")
                    ax.set_xlabel("Timestep")
                    ax.set_ylabel(f"Action Component {i + 1}")

            plt.tight_layout()
            plt.show()

        if self.plot_states:
            # Plot the states
            states_np = torch.tensor(np.array(all_states), dtype=torch.float32).detach().cpu().numpy()
            timesteps = states_np.shape[0]
            state_dim = states_np.shape[1]
            time = range(timesteps)

            # Plot each component of the state vector
            fig, ax = plt.subplots(state_dim, 1, figsize=(10, 8))

            for i in range(state_dim):
                if isinstance(ax, np.ndarray):
                    ax[i].plot(time, states_np[:, i])
                    ax[i].set_title(f"State Component {i + 1} Over Time")
                    ax[i].set_xlabel("Timestep")
                    ax[i].set_ylabel(f"State Component {i + 1}")
                else:  # In case there exists only one state dimension
                    ax.plot(time, states_np[:, i])
                    ax.set_title(f"State Component {i + 1} Over Time")
                    ax.set_xlabel("Timestep")
                    ax.set_ylabel(f"State Component {i + 1}")
            plt.tight_layout()
            plt.show()





