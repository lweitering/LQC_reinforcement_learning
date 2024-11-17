"""
This file is used to run the experiments.

@author: Lucas Weitering
"""
import itertools
import os
import shutil
import traceback

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import skewnorm

import dynamics
from algorithms.adaptive_controller import AdaptiveController, ProximalPolicyOptimization
from parameter_tuning import get_problem_formulations

# Enable LaTeX text rendering
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

# Set seaborn style for cleaner plots
sns.set_style('ticks')

# Color palette for the plots (Tab10 is commonly used for publication-quality plots)
rgblist = sns.color_palette('tab10')


# Method to run single instances with specific parameters and with additional options
def run_single():
    """
    Run a single experiment with specified configurations.

    This function sets up and executes experiments on a selected control system using specified algorithms.
    Users can adjust various run parameters such as plotting options, system dynamics, noise characteristics,
    disturbances, and nonlinearities.

    Configurable Options:
        - Select which algorithms to test (ARBMLE, STABL, PPO, LQR).
        - Choose the system dynamics (Boeing, UAV, or Cart Pole).
        - Set experiment parameters like horizon length, number of repeats, and regularization.
        - Adjust noise settings, including skewness and multiplicative scaling.
        - Introduce disturbances at specified time steps.
        - Enable or disable nonlinearity in the system dynamics.
        - Choose whether to plot actions, states, costs, and cost splits.

    The function runs the experiments for the selected algorithms, records costs, and optionally plots the results.
    It also calculates cumulative costs and prints summary statistics.

    Returns:
        None. Outputs are printed to the console, and plots are saved if plotting is enabled.
    """


    # Adapt run parameters according to needs
    plot_actions = False
    plot_states = False
    plot_costs = False
    plot_cost_split_averaged = False  # Might not work

    print_max_dif = False

    test_ARBMLE = False
    test_STABL = False
    test_PPO = True
    test_LQR = False

    # Set required system dynamics (Largest differences between LQR and ARBMLE in the first 25 indices: Boeing=7, UAV=12, CartPole=7)
    A_star, B_star, Q, R, problem = dynamics.boeing()  # Boeing system dynamics
    #A_star, B_star, Q, R, problem = dynamics.uav()  # Unmanned Aerial Vehicle
    #A_star, B_star, Q, R, problem = dynamics.cart_pole()  # Cart Pole / Inverted Pendulum

    n, m = B_star.shape

    horizon = 200  # length of each experiment
    num_repeats = 200  # number of repetitions
    reg = 1e-4  # Regularization
    T_init = 30  # Initial time step for stabilizing controller

    start_at = 0  # To check from specific seeds

    # Run Parameters
    skewness = 0  # 0 is standard case
    noise_multiplicator = 1.0  # 1.0 is standard case
    disturbance_step = False  # True if disturbance is to be inserted, else False
    disturbance_time = 100  # Time for the disturbance insertion
    nonlinearity = False  # True if nonlinear case, else False
    nonlinear_case = "x_sin(x)"  # "x_squared", "x_sin(x)", "x_log(x)"

    if test_ARBMLE: cost_ARBMLE = np.zeros((num_repeats, horizon + T_init))
    if test_STABL: cost_STABL = np.zeros((num_repeats, horizon + T_init))
    if test_PPO: cost_PPO = np.zeros((num_repeats, horizon + T_init))
    if test_LQR: cost_LQR = np.zeros((num_repeats, horizon + T_init))

    if plot_cost_split_averaged:
        if test_ARBMLE:
            input_cost_ARBMLE = np.zeros((num_repeats, horizon + T_init))
            state_cost_ARBMLE = np.zeros((num_repeats, horizon + T_init))
        if test_STABL:
            input_cost_STABL = np.zeros((num_repeats, horizon + T_init))
            state_cost_STABL = np.zeros((num_repeats, horizon + T_init))
        if test_PPO:
            input_cost_PPO = np.zeros((num_repeats, horizon + T_init))
            state_cost_PPO = np.zeros((num_repeats, horizon + T_init))
        if test_LQR:
            input_cost_LQR = np.zeros((num_repeats, horizon + T_init))
            state_cost_LQR = np.zeros((num_repeats, horizon + T_init))

    for i in range(num_repeats):
        i += start_at
        print("Experiment run {}/{}".format(i + 1, num_repeats))
        np.random.seed(i)

        # Random noise generation for input and state evolution
        input_noise = np.random.normal(0, size=(m, T_init))
        if skewness == 0:
            noise = np.random.normal(0, size=(n, horizon + T_init))
        else:
            omega = 1.032  # scale
            noise = generate_skewed_noise(n, horizon + T_init, skewness, omega, i)

        if problem == "CART_POLE":
            input_noise *= 10

        i -= start_at

        # Multiply Noise (if required)
        noise = noise * noise_multiplicator
        input_noise = input_noise * noise_multiplicator

        # Apply disturbance if required
        if disturbance_step:
            if problem == "CART_POLE":
                noise[:, disturbance_time] += [0, 0, 5, 5]  # Changing angle of the pole and angular velocity
            elif problem == "BOEING":
                noise[:, disturbance_time] += [0, 0, 8, 15]  # Changing the angle and angular velocity of the plane
            else:
                noise[:, disturbance_time] += [3, 1, 3,
                                               1]  # Adding 3 to the position on the plane, and 1 to the respective velocity

        # Instantiate new controllers with explicit hyperparameters for each algorithm
        env_ARBMLE = AdaptiveController(A_star, B_star, Q, R, horizon, T_init, reg, 'ARBMLE', cur_seed=i,
                                        num_restarts=5, max_iters=500, step_size=0.025, rel_tol=0.0001,
                                        delta=0.001, bias=0.01, S=80, plot_actions=plot_actions,
                                        plot_costs=plot_costs, plot_states=plot_states, nonlinearity=nonlinearity,
                                        nonlinear_case=nonlinear_case, problem_name_nonlin=problem)

        env_STABL = AdaptiveController(A_star, B_star, Q, R, horizon, T_init, reg, 'STABL', cur_seed=i,
                                       num_restarts=5, max_iters=500, step_size=0.005, rel_tol=0.0001,
                                       delta=0.01, bias=0.01, T_w=10, sigma_w=2, S=50,
                                       plot_actions=plot_actions, plot_states=plot_states, plot_costs=plot_costs,
                                       nonlinearity=nonlinearity, nonlinear_case=nonlinear_case,
                                       problem_name_nonlin=problem)

        env_PPO = ProximalPolicyOptimization(A_star, B_star, Q, R, horizon, T_init, reg, 'PPO', cur_seed=i,
                                             gamma=1.0, lambda_=1.0, epsilon=0.1, lr_policy=0.005,
                                             lr_value=0.01, std_clamp_max=0.7, update_steps=5,
                                             plot_actions=plot_actions, plot_states=plot_states, plot_costs=plot_costs,
                                             nonlinearity=nonlinearity, nonlinear_case=nonlinear_case,
                                             problem_name_nonlin=problem)

        env_LQR = AdaptiveController(A_star, B_star, Q, R, horizon, T_init, reg, 'LQR', cur_seed=i,
                                     plot_actions=plot_actions, plot_costs=plot_costs,
                                     plot_states=plot_states, nonlinearity=nonlinearity,
                                     nonlinear_case=nonlinear_case, problem_name_nonlin=problem)

        # Run experiments
        if test_ARBMLE:
            print("ARBMLE")
            env_ARBMLE.run_experiment(noise, input_noise, reg)
        if test_STABL:
            print("STABL")
            env_STABL.run_experiment(noise, input_noise, reg)
        if test_PPO:
            print("PPO")
            env_PPO.run_experiment(noise, input_noise, reg)
        if test_LQR:
            print("LQR")
            env_LQR.run_optimal(noise, input_noise)

        # Store the costs for each run
        if test_ARBMLE: cost_ARBMLE[i, :] = env_ARBMLE.costs
        if test_STABL: cost_STABL[i, :] = env_STABL.costs
        if test_PPO: cost_PPO[i, :] = env_PPO.costs
        if test_LQR: cost_LQR[i, :] = env_LQR.costs

        # Record cost split for each run
        if plot_cost_split_averaged:
            if test_ARBMLE:
                input_cost_ARBMLE[i, :] = env_ARBMLE.input_costs_recording
                state_cost_ARBMLE[i, :] = env_ARBMLE.state_costs_recording
                print(f"Size of input costs ARBMLE: {input_cost_ARBMLE.shape}")
                print(f"Size of state costs ARBMLE: {state_cost_ARBMLE.shape}")
            if test_STABL:
                input_cost_STABL[i, :] = env_STABL.input_costs_recording
                state_cost_STABL[i, :] = env_STABL.state_costs_recording
            if test_PPO:
                input_cost_PPO[i, :] = env_PPO.input_costs_recording
                state_cost_PPO[i, :] = env_PPO.state_costs_recording
            if test_LQR:
                input_cost_LQR[i, :] = env_LQR.input_costs_recording
                state_cost_LQR[i, :] = env_LQR.state_costs_recording

    if plot_cost_split_averaged:
        timesteps = range(horizon)
        if test_ARBMLE:
            plot_cost_parts_and_total(timesteps, state_cost_ARBMLE[:,T_init-1:], input_cost_ARBMLE[:,T_init-1:],
                                      cost_ARBMLE[:,T_init-1:], problem, "ARBMLE")
        if test_STABL:
            plot_cost_parts_and_total(timesteps, state_cost_STABL[:,T_init-1:], input_cost_STABL[:,T_init-1:],
                                      cost_STABL[:,T_init-1:], problem, "STABL")
        if test_PPO:
            plot_cost_parts_and_total(timesteps, state_cost_PPO[:,T_init-1:], input_cost_PPO[:,T_init-1:],
                                      cost_PPO[:,T_init-1:], problem, "PPO")
        if test_LQR:
            plot_cost_parts_and_total(timesteps, state_cost_LQR[:,T_init-1:], input_cost_LQR[:,T_init-1:],
                                      cost_LQR[:,T_init-1:], problem, "LQR")
        print(f"Plotted and saved cost parts and total costs!")

    # After the loop where the costs are stored
    if test_ARBMLE and test_LQR and print_max_dif:
        # Calculate cumulative costs for each algorithm if not already done
        cum_cost_ARBMLE = np.cumsum(cost_ARBMLE[:, T_init:-1], axis=1)
        cum_cost_LQR = np.cumsum(cost_LQR[:, T_init:-1], axis=1)

        # Calculate the difference between ARBMLE and LQR cumulative costs for each run
        # We are interested in the maximum difference where ARBMLE is better (i.e., ARBMLE - LQR > 0)
        diff = cum_cost_LQR[:, -1] - cum_cost_ARBMLE[:, -1]  # LQR - ARBMLE (positive when ARBMLE is better)

        # Find the run number with the maximum positive difference
        max_diff_idx = np.argmax(diff)

        # Get the costs for that run
        max_ARBMLE_cost = cum_cost_ARBMLE[max_diff_idx, -1]
        max_LQR_cost = cum_cost_LQR[max_diff_idx, -1]

        # Print the run number and the corresponding costs
        print(f"Run {max_diff_idx}: Maximum difference where ARBMLE is better for problem {problem}.")
        print(f"ARBMLE cumulative cost: {max_ARBMLE_cost}")
        print(f"LQR cumulative cost: {max_LQR_cost}")

    # Calculate and save cumulative costs for each algorithm
    if test_ARBMLE: cum_cost_ARBMLE = np.cumsum(cost_ARBMLE[:, T_init:-1], axis=1)
    if test_STABL: cum_cost_STABL = np.cumsum(cost_STABL[:, T_init:-1], axis=1)
    if test_PPO: cum_cost_PPO = np.cumsum(cost_PPO[:, T_init:-1], axis=1)
    if test_LQR: cum_cost_LQR = np.cumsum(cost_LQR[:, T_init:-1], axis=1)

    if test_ARBMLE: avg_cost_ARBMLE = np.mean(cum_cost_ARBMLE, axis=0)
    if test_STABL: avg_cost_STABL = np.mean(cum_cost_STABL, axis=0)
    if test_PPO: avg_cost_PPO = np.mean(cum_cost_PPO, axis=0)
    if test_LQR: avg_cost_LQR = np.mean(cum_cost_LQR, axis=0)

    if test_ARBMLE: print('ARBMLE', avg_cost_ARBMLE[-1])
    if test_STABL: print('STABL', avg_cost_STABL[-1])
    if test_PPO: print('PPO', avg_cost_PPO[-1])
    if test_LQR: print('LQR', avg_cost_LQR[-1])


def plot_cost_parts_and_total(timesteps, state_costs, input_costs, total_costs, problem_name, algorithm_name):
    """
    Plots the state cost, input cost, and total cost in both stacked and combined formats.

    :param timesteps: List or array of timesteps.
    :param state_costs: List or array of state costs over time.
    :param input_costs: List or array of input costs over time.
    :param total_costs: List or array of total costs over time.
    :param problem_name: Name of the problem setting (used in plot titles).
    :param algorithm_name: Name of the algorithm (used in plot titles).
    """

    mean_state_costs, median_state_costs, low_state_costs, high_state_costs = get_percentile_confidence(state_costs)
    mean_input_costs, median_input_costs, low_input_costs, high_input_costs = get_percentile_confidence(input_costs)
    mean_costs, median_costs, low_costs, high_costs = get_percentile_confidence(total_costs)

    # 1. Create stacked subplots for state, input, and total costs
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Plot state cost
    print(f"Shape timesteps: {np.array(timesteps).shape}, Mean State Costs shape: {mean_state_costs.shape}")
    ax[0].plot(timesteps, mean_state_costs, label="State Cost $(x * Q * x)$", color='b')
    ax[0].plot(timesteps, median_state_costs, color='b', linestyle='--')
    ax[0].fill_between(np.arange(len(timesteps)), low_state_costs, high_state_costs, color='b', alpha=0.1)
    ax[0].set_title(f"State Cost for {problem_name} {algorithm_name}")
    ax[0].set_ylabel("Cost")
    ax[0].legend()

    # Plot input cost
    ax[1].plot(timesteps, mean_input_costs, label="Input Cost $(u * R * u)$", color='g')
    ax[1].plot(timesteps, median_input_costs, color='g', linestyle='--')
    ax[1].fill_between(np.arange(len(timesteps)), low_input_costs, high_input_costs, color='g', alpha=0.1)
    ax[1].set_title(f"Input Cost for {problem_name} {algorithm_name}")
    ax[1].set_ylabel("Cost")
    ax[1].legend()

    # Plot total cost
    low_costs[-1] = low_costs[-2]
    ax[2].plot(timesteps, mean_costs, label="Total Cost $c(x,u)$", color='r')
    ax[2].plot(timesteps, median_costs, color='r', linestyle='--')
    ax[2].fill_between(np.arange(len(timesteps)), low_costs, high_costs, color='r', alpha=0.1)
    ax[2].set_title(f"Total Cost for {problem_name} {algorithm_name}")
    ax[2].set_xlabel("Timestep")
    ax[2].set_ylabel("Cost")
    ax[2].legend()

    plt.tight_layout()

    # Create directory if it doesn't exist
    output_dir = os.path.join("plots", "cost_analysis_average")
    os.makedirs(output_dir, exist_ok=True)

    # Save the stacked version
    filename_stacked = f"cost_breakdown_stacked_{problem_name}_{algorithm_name}.pdf"
    file_path_stacked = os.path.join(output_dir, filename_stacked)
    plt.savefig(file_path_stacked)
    plt.close()

    print(f"Stacked cost breakdown plot saved at: {file_path_stacked}")

    # 2. Create combined plot with all curves on the same graph
    fig = plt.figure()
    plt.plot(timesteps, mean_state_costs, label="State Cost $(x * Q * x)$", color='b')
    plt.plot(timesteps, median_state_costs, color='b', linestyle='--')
    plt.fill_between(np.arange(len(timesteps)), low_state_costs, high_state_costs, color='b', alpha=0.1)

    plt.plot(timesteps, mean_input_costs, label="Input Cost $(u * R * u)$", color='g')
    plt.plot(timesteps, median_input_costs, color='g', linestyle='--')
    plt.fill_between(np.arange(len(timesteps)), low_input_costs, high_input_costs, color='g', alpha=0.1)

    plt.plot(timesteps, mean_costs, label="Total Cost $c(x,u)$", color='r')
    plt.plot(timesteps, median_costs, color='r', linestyle='--')
    plt.fill_between(np.arange(len(timesteps)), low_costs, high_costs, color='r', alpha=0.1)

    plt.title(f"Cost Breakdown {problem_name} {algorithm_name}")
    plt.xlabel("Timestep")
    plt.ylabel("Cost")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()

    # Save the combined version
    filename_combined = f"cost_breakdown_{problem_name}_{algorithm_name}.pdf"
    file_path_combined = os.path.join(output_dir, filename_combined)
    fig.savefig(file_path_combined)
    plt.close()

    print(f"Combined cost breakdown plot saved at: {file_path_combined}")

    # Optionally, print cumulative sums
    sum_state_costs = np.cumsum(state_costs)
    sum_input_costs = np.cumsum(input_costs)
    sum_total_costs = np.cumsum(total_costs)

    print(
        f"For algorithm: {algorithm_name} and problem: {problem_name}\n"
        f"State cost sum: {sum_state_costs[-1]}\n"
        f"Input cost sum: {sum_input_costs[-1]}\n"
        f"Total cost sum: {sum_total_costs[-1]}"
    )


# Function to calculate the theoretical mean of a skewed normal distribution
def skewnorm_mean(alpha, omega, xi):
    """
    Calculate the theoretical mean of a skew-normal distribution.

    Parameters:
        alpha (float): Skewness parameter.
        omega (float): Scale parameter.
        xi (float): Location parameter.

    Returns:
        float: The mean of the skew-normal distribution.
    """
    delta = alpha / np.sqrt(1 + alpha ** 2)
    return xi + omega * delta * np.sqrt(2 / np.pi)


# Function to generate skew-normal distribution
def generate_skewed_noise(m, T, alpha, omega, seed=None):
    """
    Generate skew-normal noise with zero mean for simulations.

    Parameters:
        m (int): Number of dimensions.
        T (int): Number of time steps.
        alpha (float): Skewness parameter.
        omega (float): Scale parameter.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Skew-normal noise array of shape (m, T) with zero mean.
    """
    # Calculate the theoretical mean of the distribution
    theoretical_mean = skewnorm_mean(alpha, omega, 0)
    # Adjust location parameter xi to shift the mean to zero
    xi = -theoretical_mean

    # Set random state for reproducibility
    rng = np.random.default_rng(seed)

    noise = skewnorm.rvs(alpha, loc=xi, scale=omega, size=(m, T), random_state=rng)
    return noise


def start_experiment(horizon, num_repeats, reg, T_init, noise_multiplicator=1.0,
                     disturbance_step=False, disturbance_time=None, skewness=0,
                     nonlinearity=False, nonlinear_case=None,
                     plot_actions=False, plot_states=False,
                     save_file_name="", run_optimal=False):
    """
    Run experiments across multiple control systems with specified configurations.

    This function executes experiments on predefined control systems (Boeing, UAV, CartPole)
    using selected algorithms under various settings such as noise levels, disturbances,
    skewness, and nonlinearities. It saves the cumulative costs for later analysis.

    Parameters:
        horizon (int): Length of each experiment run.
        num_repeats (int): Number of repetitions for each experiment.
        reg (float): Regularization parameter for the controllers.
        T_init (int): Initial time steps using a stabilizing controller.
        noise_multiplicator (float, optional): Multiplier to scale noise magnitude.
        disturbance_step (bool, optional): Whether to introduce a disturbance.
        disturbance_time (int, optional): Time step at which to apply the disturbance.
        skewness (float, optional): Skewness parameter for the noise distribution.
        nonlinearity (bool, optional): Whether to include nonlinearity in the system.
        nonlinear_case (str, optional): Type of nonlinearity to apply ("x_squared", "x_sin(x)", "x_log(x)").
        plot_actions (bool, optional): Whether to plot actions during the experiment.
        plot_states (bool, optional): Whether to plot states during the experiment.
        save_file_name (str, optional): Prefix for saving result files.
        run_optimal (bool, optional): If True, runs only the optimal controller (LQR).

    Returns:
        None. Results are saved to files, and progress is printed to the console.
    """

    problems = get_problem_formulations()

    try:
        # Loop over all problems and store results
        for problem_name, (A_star, B_star, Q, R, _) in problems.items():
            if run_optimal:
                print(f"Running with LQR (Optimal Controller) on {save_file_name} setting")
            else:
                print(f"Running {save_file_name} setting")

            n, m = B_star.shape

            if run_optimal:
                cost_OPT = np.zeros((num_repeats, horizon + T_init))
            else:
                cost_ARBMLE = np.zeros((num_repeats, horizon + T_init))
                cost_STABL = np.zeros((num_repeats, horizon + T_init))
                cost_PPO = np.zeros((num_repeats, horizon + T_init))

            current_algorithm = ""  # Debug
            try:
                for i in range(num_repeats):
                    if i % 10 == 0:
                        print("Experiment run {}/{} on {}".format(i + 1, num_repeats, problem_name))
                    np.random.seed(i)

                    # Random noise generation for input and state evolution
                    input_noise = np.random.normal(0, size=(m, T_init))
                    if skewness == 0:
                        noise = np.random.normal(0, size=(n, horizon + T_init))
                    else:
                        omega = 1.032  # scale
                        noise = generate_skewed_noise(n, horizon + T_init, skewness, omega, i)

                    if problem_name == "CartPole":
                        input_noise *= 10

                    # Multiply Noise (if required)
                    noise = noise * noise_multiplicator
                    input_noise = input_noise * noise_multiplicator

                    # Apply disturbance if required
                    if disturbance_step:
                        if problem_name == "CartPole":
                            noise[:, disturbance_time] += [0, 0, 5,
                                                           5]  # Changing angle of the pole and angular velocity
                        elif problem_name == "Boeing":
                            noise[:, disturbance_time] += [0, 0, 8,
                                                           15]  # Changing the angle and angular velocity of the plane
                        else:
                            noise[:, disturbance_time] += [3, 1, 3,
                                                           1]  # Adding 3 to the position on the plane, and 1 to the respective velocity

                    # Instantiate new controllers with explicit hyperparameters for each algorithm
                    env_ARBMLE = AdaptiveController(A_star, B_star, Q, R, horizon, T_init, reg, 'ARBMLE', cur_seed=i,
                                                    num_restarts=5, max_iters=500, step_size=0.025, rel_tol=0.0001,
                                                    delta=0.001, bias=0.01, S=80, plot_actions=plot_actions,
                                                    plot_states=plot_states, nonlinearity=nonlinearity,
                                                    nonlinear_case=nonlinear_case, problem_name_nonlin=problem_name)

                    env_STABL = AdaptiveController(A_star, B_star, Q, R, horizon, T_init, reg, 'STABL', cur_seed=i,
                                                   num_restarts=5, max_iters=500, step_size=0.005, rel_tol=0.0001,
                                                   delta=0.01, bias=0.01, T_w=10, sigma_w=2, S=50,
                                                   plot_actions=plot_actions, plot_states=plot_states,
                                                   nonlinearity=nonlinearity, nonlinear_case=nonlinear_case,
                                                   problem_name_nonlin=problem_name)

                    env_PPO = ProximalPolicyOptimization(A_star, B_star, Q, R, horizon, T_init, reg, 'PPO', cur_seed=i,
                                                         gamma=1.0, lambda_=1.0, epsilon=0.1, lr_policy=0.005,
                                                         lr_value=0.01, std_clamp_max=0.7, update_steps=5,
                                                         plot_actions=plot_actions, plot_states=plot_states,
                                                         nonlinearity=nonlinearity, nonlinear_case=nonlinear_case,
                                                         problem_name_nonlin=problem_name)

                    # Run experiments
                    if run_optimal:
                        current_algorithm = "LQR"
                        env_ARBMLE.run_optimal(noise, input_noise)
                    else:
                        current_algorithm = "ARBMLE"
                        env_ARBMLE.run_experiment(noise, input_noise, reg)
                        current_algorithm = "STABL"
                        env_STABL.run_experiment(noise, input_noise, reg)
                        current_algorithm = "PPO"
                        env_PPO.run_experiment(noise, input_noise, reg)

                    # Store the costs for each run
                    if run_optimal:
                        cost_OPT[i, :] = env_ARBMLE.costs
                    else:
                        cost_ARBMLE[i, :] = env_ARBMLE.costs
                        cost_STABL[i, :] = env_STABL.costs
                        cost_PPO[i, :] = env_PPO.costs
            except Exception as e:
                print(f"Experiment with configuration: noise_multiplicator={noise_multiplicator}, "
                      f"disturbance_step={disturbance_step}, disturbance_time={disturbance_time}, "
                      f"skewness={skewness}, nonlinearity={nonlinearity}, and nonlinear_case={nonlinear_case} "
                      f"has failed!")
                print(f"Failed for problem_name={problem_name} with algorithm={current_algorithm}")
                print(f"Error: {e}")
                traceback.print_exc()

            # Calculate and save cumulative costs for each algorithm
            if run_optimal:
                np.save(f'cum_cost_of_t/OPTIMAL_{save_file_name}_{problem_name}',
                        np.cumsum(cost_OPT[:, T_init-1:-1], axis=1))
            else:
                cum_cost_ARBMLE = np.cumsum(cost_ARBMLE[:, T_init-1:-1], axis=1)
                cum_cost_STABL = np.cumsum(cost_STABL[:, T_init-1:-1], axis=1)
                cum_cost_PPO = np.cumsum(cost_PPO[:, T_init-1:-1], axis=1)

                np.save(f'cum_cost_of_t/ARBMLE_{save_file_name}_{problem_name}', cum_cost_ARBMLE)
                print(f"Last cumulative cost for ARBMLE (run_idx = 0): {cum_cost_ARBMLE[0, -1]}")
                np.save(f'cum_cost_of_t/STABL_{save_file_name}_{problem_name}', cum_cost_STABL)
                print(f"Last cumulative cost for STABL (run_idx = 0): {cum_cost_STABL[0, -1]}")
                np.save(f'cum_cost_of_t/PPO_{save_file_name}_{problem_name}', cum_cost_PPO)
                print(f"Last cumulative cost for PPO (run_idx = 0): {cum_cost_PPO[0, -1]}")
    except Exception as e:
        print(f"Experiment with configuration: noise_multiplicator={noise_multiplicator}, "
              f"disturbance_step={disturbance_step}, disturbance_time={disturbance_time}, "
              f"skewness={skewness}, nonlinearity={nonlinearity}, and nonlinear_case={nonlinear_case} has failed!")
        print(f"Error: {e}")
        traceback.print_exc()


def run_all_tests(run_optimal=False):
    """
    Run a comprehensive suite of experiments under various configurations.

    This function automates the execution of multiple experiments by calling `start_experiment` with different parameters.
    It tests the performance of controllers under a variety of conditions, including varying noise levels,
    skewness in noise distribution, disturbances at specific time steps, and introducing nonlinearities in the system.

    Parameters:
        run_optimal (bool): If True, runs only the optimal controller (LQR) for all tests.
                            If False, runs the specified adaptive algorithms.

    Experiments conducted:
        1. Standard variant:
            - Standard noise (normal distribution with mean 0 and standard deviation 1).
            - No disturbances.
            - No skewness.
            - No nonlinearities.
            - `save_file_name="standard"`

        1.5. Standard variant with extended horizon:
            - Same as the standard variant but with a longer horizon
            - `save_file_name="standard_longrun"`

        2. Different noise levels:
            - Noise multiplier of 0.1 (reduced noise).
                - `save_file_name="noise01"`
            - Noise multiplier of 1.0 (standard noise).
                - `save_file_name="noise1"` (same as the standard variant)
            - Noise multiplier of 10.0 (increased noise).
                - `save_file_name="noise10"`

        3. Experiments with skewness at different noise levels:
            - Skewness of -10 with noise multiplier of 1.0.
                - `save_file_name="noise1skewMin10"`
            - Skewness of +10 with noise multiplier of 1.0.
                - `save_file_name="noise1skewPlu10"`
            - Skewness of +10 with noise multiplier of 10.0.
                - `save_file_name="noise10skewPlu10"`
            - Skewness of -10 with noise multiplier of 10.0.
                - `save_file_name="noise10skewMin10"`

        4. Experiment with a disturbance:
            - Standard noise.
            - A disturbance introduced at time step `t = 70`.
                - `save_file_name="disturbance"`

        5. Experiments with nonlinearities:
            - Nonlinear case "x_squared".
                - `save_file_name="nonlinear_x_squared"`
            - Nonlinear case "x_sin(x)".
                - `save_file_name="nonlinear_x_sin(x)"`
            - Nonlinear case "x_log(x)".
                - `save_file_name="nonlinear_x_log(x)"`
            - Note: The number of repeats (`num_repeats`) is reduced to 5 for nonlinear experiments,
              as successful control is not guaranteed, and fewer runs suffice for analysis.

    Notes:
        - The `horizon`, `num_repeats`, `reg`, and `T_init` parameters are set at the beginning and can be adjusted as needed.
        - The `save_file_name` parameter helps in identifying and saving the results for each experiment configuration.
        - The `run_optimal` flag allows switching between running the optimal controller and the adaptive algorithms.

    Returns:
        None. The function executes the experiments and saves the cumulative costs for analysis.
    """
    # Test Parameters
    horizon = 200  # length of each experiment
    num_repeats = 200  # number of repetitions
    reg = 1e-4
    T_init = 30  # Initial time step for stabilizing controller

    # Running experiments based on chapter "Methodology"
    # 1. Standard variant (standard noise, no disturbance, no skewness, no nonlinearity)
    start_experiment(horizon, num_repeats, reg, T_init, save_file_name="standard", run_optimal=run_optimal)

    # 1.5 Standard variant with 5x time horizon
    pre_horizon = horizon
    horizon = 10000
    start_experiment(horizon, num_repeats, reg, T_init, save_file_name="standard_longrun", run_optimal=run_optimal)
    horizon = pre_horizon

    # 2. Different noise levels (0.1, 1, 10)
    start_experiment(horizon, num_repeats, reg, T_init, noise_multiplicator=0.1, save_file_name="noise01", run_optimal=run_optimal)
    start_experiment(horizon, num_repeats, reg, T_init, noise_multiplicator=1.0, save_file_name="noise1", run_optimal=run_optimal)  # Same as standard
    start_experiment(horizon, num_repeats, reg, T_init, noise_multiplicator=10.0, save_file_name="noise10",
                     run_optimal=run_optimal)

    # 3. Experiments with skewness at different noise levels
    start_experiment(horizon, num_repeats, reg, T_init, noise_multiplicator=1.0,  #
                     skewness=-10, save_file_name="noise1skewMin10", run_optimal=run_optimal)
    start_experiment(horizon, num_repeats, reg, T_init, noise_multiplicator=1.0,  #
                     skewness=10, save_file_name="noise1skewPlu10", run_optimal=run_optimal)
    start_experiment(horizon, num_repeats, reg, T_init, noise_multiplicator=10.0,
                     skewness=10, save_file_name="noise10skewPlu10", run_optimal=run_optimal)
    start_experiment(horizon, num_repeats, reg, T_init, noise_multiplicator=10.0,
                     skewness=-10, save_file_name="noise10skewMin10", run_optimal=run_optimal)

    # 4. Experiment with standard noise and a disturbance at time step t = 100
    start_experiment(horizon, num_repeats, reg, T_init, disturbance_step=True,  #
                     disturbance_time=100, save_file_name="disturbance", run_optimal=run_optimal)

    # 5. Experiment with normal noise and nonlinear
    # "x_squared", "x_sin(x)", "x_log(x)"
    num_repeats = 5  # In the nonlinear setting, we do not assume, that the algorithm is able to successfully control the control system. 5 Runs are required for plotting.
    start_experiment(horizon, num_repeats, reg, T_init, nonlinearity=True, save_file_name="nonlinear_x_squared",
                     nonlinear_case="x_squared", run_optimal=run_optimal)
    start_experiment(horizon, num_repeats, reg, T_init, nonlinearity=True, save_file_name="nonlinear_x_sin(x)",
                     nonlinear_case="x_sin(x)", run_optimal=run_optimal)
    start_experiment(horizon, num_repeats, reg, T_init, nonlinearity=True, save_file_name="nonlinear_x_log(x)",
                     nonlinear_case="x_log(x)", run_optimal=run_optimal)


def get_mean(data):
    """
    Calculate the mean, lower, and upper bound (mean +/- std) for the data.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    low = mean - std
    high = mean + std
    return mean, low, high


def get_percentile_confidence(data, lower_percentile=10, upper_percentile=90):
    """
    Calculate the mean, median, lower, and upper bound using percentiles for the data.
    """
    mean = np.mean(data, axis=0)
    median = np.median(data, axis=0)
    low = np.percentile(data, lower_percentile, axis=0)
    high = np.percentile(data, upper_percentile, axis=0)
    return mean, median, low, high


def generate_cum_cost_plot(save_file_name, problem_name):
    # Load the cost data for the algorithms and LQR controller
    cost_ARBMLE = np.load(f'costs_of_t/Cost_ARBMLE_{save_file_name}_{problem_name}.npy')
    cost_STABL = np.load(f'costs_of_t/Cost_STABL_{save_file_name}_{problem_name}.npy')
    cost_PPO = np.load(f'costs_of_t/Cost_PPO_{save_file_name}_{problem_name}.npy')
    cost_OPTIMAL = np.load(f'costs_of_t/Cost_OPTIMAL_{save_file_name}_{problem_name}.npy')

    # Compute cumulative regret (difference in cumulative cost between each algorithm and OPTIMAL)
    cum_regret_ARBMLE = np.cumsum(cost_ARBMLE, axis=1)
    cum_regret_STABL = np.cumsum(cost_STABL, axis=1)
    cum_regret_PPO = np.cumsum(cost_PPO, axis=1)
    cum_regret_OPTIMAL = np.cumsum(cost_OPTIMAL, axis=1)

    # Get the mean, lower, and upper bounds for cumulative regret of each algorithm
    mean_ARBMLE, median_ARBMLE, low_ARBMLE, high_ARBMLE = get_percentile_confidence(cum_regret_ARBMLE)
    mean_STABL, median_STABL, low_STABL, high_STABL = get_percentile_confidence(cum_regret_STABL)
    mean_PPO, median_PPO, low_PPO, high_PPO = get_percentile_confidence(cum_regret_PPO)
    mean_OPTIMAL, median_OPTIMAL, low_OPTIMAL, high_OPTIMAL = get_percentile_confidence(cum_regret_OPTIMAL)

    # Create the plot
    fig = plt.figure()

    # Use a symlog scale for the y-axis (logarithmic for large values, linear for small values)
    linthresh = 1  # The threshold below which it switches to a linear scale

    # Plot cumulative cost for ARBMLE
    plt.plot(range(len(mean_ARBMLE)), mean_ARBMLE, color=rgblist[0], label='ARBMLE')
    plt.plot(range(len(mean_ARBMLE)), median_ARBMLE, color=rgblist[0], linestyle='--')
    plt.fill_between(np.arange(len(mean_ARBMLE)), low_ARBMLE, high_ARBMLE, color=rgblist[0], alpha=0.1)

    # Plot cumulative cost for STABL
    plt.plot(range(len(mean_STABL)), mean_STABL, color=rgblist[1], label='STABL')
    plt.plot(range(len(mean_STABL)), median_STABL, color=rgblist[1], linestyle='--')
    plt.fill_between(np.arange(len(mean_STABL)), low_STABL, high_STABL, color=rgblist[1], alpha=0.1)

    # Plot cumulative cost for PPO
    plt.plot(range(len(mean_PPO)), mean_PPO, color=rgblist[2], label='PPO')
    plt.plot(range(len(mean_PPO)), median_PPO, color=rgblist[2], linestyle='--')
    plt.fill_between(np.arange(len(mean_PPO)), low_PPO, high_PPO, color=rgblist[2], alpha=0.1)

    # Plot cumulative cost for OPTIMAL
    plt.plot(range(len(mean_OPTIMAL)), mean_OPTIMAL, color=rgblist[3], label='LQR')
    plt.plot(range(len(mean_OPTIMAL)), median_OPTIMAL, color=rgblist[3], linestyle='--')
    plt.fill_between(np.arange(len(mean_OPTIMAL)), low_OPTIMAL, high_OPTIMAL, color=rgblist[3], alpha=0.1)

    # Set y-axis to symlog scale
    plt.yscale('symlog', linthresh=linthresh)

    # Add labels and title
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.ylabel('Cumulative Cost', fontsize=16)
    plt.xlabel('Time step $t$', fontsize=16)

    # Ensure the directory exists for saving the plots
    plot_dir = 'plots/cum_cost/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Save the plot
    fig.savefig(f'{plot_dir}/cum_cost_{save_file_name}_{problem_name}.pdf', bbox_inches='tight')
    plt.close()


def cumulative_cost_to_cost(filename):
    """
    Convert cumulative costs to single-step costs and save them.

    Parameters:
        filename (str): Base name of the cumulative cost file (without path or extension).

    The function reads cumulative costs from 'cum_cost_of_t/{filename}.npy',
    computes the single-step costs, and saves them to 'costs_of_t/Cost_{filename}.npy'.
    """
    try:
        # Ensure the directory exists
        output_dir = "costs_of_t"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load the cumulative costs from the file
        cum_costs = np.load(f'cum_cost_of_t/{filename}.npy')

        # Calculate the single-step costs by reversing the cumulative sum
        step_costs = np.diff(cum_costs, axis=1, prepend=0)
        # print(f"Step costs for {filename}: \n{step_costs}")

        # Save the decomposed single-step costs to the new directory with the prefix "Cost_"
        new_filename = os.path.join(output_dir, f'Cost_{filename}.npy')
        np.save(new_filename, step_costs)

        print(f"Single-step costs saved successfully to {new_filename}")

    except Exception as e:
        print(f"Failed to process file: {filename}")
        print(f"Error: {e}")


def move_files(filename):
    try:
        # Ensure the target directory exists
        target_dir = "plots"
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        # Construct the source and destination paths
        source_path = f"{filename}.pdf"
        destination_path = os.path.join(target_dir, f"{filename}.pdf")

        # Move the file to the target directory
        shutil.move(source_path, destination_path)

        print(f"File {filename}.npy successfully moved to {target_dir}")

    except Exception as e:
        print(f"Failed to move file: {filename}")
        print(f"Error: {e}")


def plot_cost_of_t(save_file_name, problem_name):
    # Load the cost data for the algorithms
    cost_ARBMLE = np.load(f'costs_of_t/Cost_ARBMLE_{save_file_name}_{problem_name}.npy')
    cost_STABL = np.load(f'costs_of_t/Cost_STABL_{save_file_name}_{problem_name}.npy')
    cost_PPO = np.load(f'costs_of_t/Cost_PPO_{save_file_name}_{problem_name}.npy')
    cost_OPTIMAL = np.load(f'costs_of_t/Cost_OPTIMAL_{save_file_name}_{problem_name}.npy')

    # Get the mean, lower, and upper bounds for each algorithm
    mean_ARBMLE, median_ARBMLE, low_ARBMLE, high_ARBMLE = get_percentile_confidence(cost_ARBMLE)
    mean_STABL, median_STABL, low_STABL, high_STABL = get_percentile_confidence(cost_STABL)
    mean_PPO, median_PPO, low_PPO, high_PPO = get_percentile_confidence(cost_PPO)
    mean_OPTIMAL, median_OPTIMAL, low_OPTIMAL, high_OPTIMAL = get_percentile_confidence(cost_OPTIMAL)

    # Create the plot
    fig = plt.figure()

    # Plot ARBMLE
    plt.plot(range(len(mean_ARBMLE)), np.log10(mean_ARBMLE), color=rgblist[0], label='ARBMLE')
    plt.plot(range(len(mean_ARBMLE)), np.log10(median_ARBMLE), color=rgblist[0], linestyle='--')
    plt.fill_between(np.arange(len(mean_ARBMLE)), np.log10(low_ARBMLE), np.log10(high_ARBMLE), color=rgblist[0],
                     alpha=0.1)

    # Plot STABL
    plt.plot(range(len(mean_STABL)), np.log10(mean_STABL), color=rgblist[1], label='STABL')
    plt.plot(range(len(mean_STABL)), np.log10(median_STABL), color=rgblist[1], linestyle='--')
    plt.fill_between(np.arange(len(mean_STABL)), np.log10(low_STABL), np.log10(high_STABL), color=rgblist[1], alpha=0.1)

    # Plot PPO
    plt.plot(range(len(mean_PPO)), np.log10(mean_PPO), color=rgblist[2], label='PPO')
    plt.plot(range(len(mean_PPO)), np.log10(median_PPO), color=rgblist[2], linestyle='--')
    plt.fill_between(np.arange(len(mean_PPO)), np.log10(low_PPO), np.log10(high_PPO), color=rgblist[2], alpha=0.1)

    # Plot OPTIMAL
    plt.plot(range(len(mean_OPTIMAL)), np.log10(mean_OPTIMAL), color=rgblist[3], label='LQR')
    plt.plot(range(len(mean_OPTIMAL)), np.log10(median_OPTIMAL), color=rgblist[3], linestyle='--')
    plt.fill_between(np.arange(len(mean_OPTIMAL)), np.log10(low_OPTIMAL), np.log10(high_OPTIMAL), color=rgblist[3],
                     alpha=0.1)

    # Add labels and title
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True)
    plt.ylabel('Logarithmic Cost $log_{10} c(t)$', fontsize=16)
    plt.xlabel('Time step $t$', fontsize=16)

    # Ensure the directory exists for saving the plots
    plot_dir = 'plots/cost_of_t/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Save the plot
    fig.savefig(f'{plot_dir}/cost_of_t_{save_file_name}_{problem_name}.pdf', bbox_inches='tight')
    plt.close()


def plot_cost_progression_for_runs(save_file_name, problem_name):
    """
    Plot the cost progression over time for multiple runs of different algorithms.

    Parameters:
        save_file_name (str): Base filename for loading cost data.
        problem_name (str): Name of the problem (e.g., "Boeing", "UAV", "CartPole").

    This function loads cost data for the specified problem and algorithms (ARBMLE, STABL, PPO, LQR),
    applies an upper cost limit to avoid extreme values, and plots the cost progression over time
    for a specified number of runs. The plots are saved in the 'plots/nonlinear_cost_of_t/' directory.
    """
    # Load the cost data for the algorithms
    cost_ARBMLE = np.load(f'costs_of_t/Cost_ARBMLE_{save_file_name}_{problem_name}.npy')
    cost_STABL = np.load(f'costs_of_t/Cost_STABL_{save_file_name}_{problem_name}.npy')
    cost_PPO = np.load(f'costs_of_t/Cost_PPO_{save_file_name}_{problem_name}.npy')
    cost_OPTIMAL = np.load(f'costs_of_t/Cost_OPTIMAL_{save_file_name}_{problem_name}.npy')

    # Define upper limit for cost to avoid plotting extreme values
    upper_limit_boeing = 10000
    upper_limit_uav = 5000
    upper_limit_cartpole = 1500000

    if problem_name == "Boeing":
        upper_limit = upper_limit_boeing
    elif problem_name == "UAV":
        upper_limit = upper_limit_uav
    elif problem_name == "CartPole":
        upper_limit = upper_limit_cartpole
    else:
        print("Unknown problem name")
        raise Exception

    # Function to smoothly exit the graph when limits are crossed
    def smooth_exit(cost_data, upper_limit):
        cost_data_smoothed = cost_data.copy()
        for i in range(cost_data.shape[0]):
            for j in range(cost_data.shape[1]):
                if cost_data_smoothed[i, j] > upper_limit:
                    # Set values exceeding the limit to NaN for smooth re-entry
                    cost_data_smoothed[i, j+1:] = np.nan
                    break
        return cost_data_smoothed

    # Apply the smooth exit function to each algorithm's cost data
    cost_ARBMLE = smooth_exit(cost_ARBMLE, upper_limit)
    cost_STABL = smooth_exit(cost_STABL, upper_limit)
    cost_PPO = smooth_exit(cost_PPO, upper_limit)
    cost_OPTIMAL = smooth_exit(cost_OPTIMAL, upper_limit)

    # Create a plot for each algorithm
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))

    # Number of runs that we are going to plot
    plot_runs = 5

    # Set y-limits for meaningful graphs
    y_lower_limit = 0
    y_upper_limit = upper_limit  # Set to the same upper limit used for filtering

    # Plot ARBMLE runs
    for i in range(plot_runs):  # Iterate over the number of runs
        ax[0, 0].plot(range(cost_ARBMLE.shape[1]), cost_ARBMLE[i], label=f'Run {i + 1}', color=rgblist[i+5])
    ax[0, 0].set_title('ARBMLE - Cost Progression')
    ax[0, 0].set_xlabel('Time step $t$')
    ax[0, 0].set_ylabel('Cost $c(t)$')
    ax[0, 0].set_ylim(y_lower_limit, y_upper_limit)  # Set y-limits
    ax[0, 0].grid(True)

    # Plot STABL runs
    for i in range(plot_runs):
        ax[0, 1].plot(range(cost_STABL.shape[1]), cost_STABL[i], label=f'Run {i + 1}', color=rgblist[i+5])
    ax[0, 1].set_title('STABL - Cost Progression')
    ax[0, 1].set_xlabel('Time step $t$')
    ax[0, 1].set_ylabel('Cost $c(t)$')
    ax[0, 1].set_ylim(y_lower_limit, y_upper_limit)
    ax[0, 1].grid(True)

    # Plot PPO runs
    for i in range(plot_runs):
        ax[1, 0].plot(range(cost_PPO.shape[1]), cost_PPO[i], label=f'Run {i + 1}', color=rgblist[i+5])
    ax[1, 0].set_title('PPO - Cost Progression')
    ax[1, 0].set_xlabel('Time step $t$')
    ax[1, 0].set_ylabel('Cost $c(t)$')
    ax[1, 0].set_ylim(y_lower_limit, y_upper_limit)
    ax[1, 0].grid(True)

    # Plot OPTIMAL runs
    for i in range(plot_runs):
        ax[1, 1].plot(range(cost_OPTIMAL.shape[1]), cost_OPTIMAL[i], label=f'Run {i + 1}', color=rgblist[i+5])
    ax[1, 1].set_title('LQR - Cost Progression')
    ax[1, 1].set_xlabel('Time step $t$')
    ax[1, 1].set_ylabel('Cost $c(t)$')
    ax[1, 1].set_ylim(y_lower_limit, y_upper_limit)
    ax[1, 1].grid(True)

    # Add a legend for one of the plots (optional, can be removed if the plot gets too cluttered)
    ax[1, 1].legend(loc='upper left', fontsize=10)

    # Ensure the directory exists for saving the plots
    plot_dir = 'plots/nonlinear_cost_of_t/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Save the plot
    fig.tight_layout()
    fig.savefig(f'{plot_dir}/cost_progression_{save_file_name}_{problem_name}.pdf', bbox_inches='tight')
    plt.close()


def plot_cost_of_t_triple(problem_name):
    noise_levels = ["noise01", "standard", "noise10"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for i, noise_level in enumerate(noise_levels):
        # Load the cost data for the algorithms
        cost_ARBMLE = np.load(f'costs_of_t/Cost_ARBMLE_{noise_level}_{problem_name}.npy')
        cost_STABL = np.load(f'costs_of_t/Cost_STABL_{noise_level}_{problem_name}.npy')
        cost_PPO = np.load(f'costs_of_t/Cost_PPO_{noise_level}_{problem_name}.npy')
        cost_OPTIMAL = np.load(f'costs_of_t/Cost_OPTIMAL_{noise_level}_{problem_name}.npy')

        # Get the mean, lower, and upper bounds for each algorithm
        mean_ARBMLE, median_ARBMLE, low_ARBMLE, high_ARBMLE = get_percentile_confidence(cost_ARBMLE)
        mean_STABL, median_STABL, low_STABL, high_STABL = get_percentile_confidence(cost_STABL)
        mean_PPO, median_PPO, low_PPO, high_PPO = get_percentile_confidence(cost_PPO)
        mean_OPTIMAL, median_OPTIMAL, low_OPTIMAL, high_OPTIMAL = get_percentile_confidence(cost_OPTIMAL)

        # Plot ARBMLE
        axs[i].plot(range(len(mean_ARBMLE)), np.log10(mean_ARBMLE), color=rgblist[0], label='ARBMLE')
        axs[i].plot(range(len(mean_ARBMLE)), np.log10(median_ARBMLE), color=rgblist[0], linestyle='--')
        axs[i].fill_between(np.arange(len(mean_ARBMLE)), np.log10(low_ARBMLE), np.log10(high_ARBMLE), color=rgblist[0],
                            alpha=0.1)

        # Plot STABL
        axs[i].plot(range(len(mean_STABL)), np.log10(mean_STABL), color=rgblist[1], label='STABL')
        axs[i].plot(range(len(mean_STABL)), np.log10(median_STABL), color=rgblist[1], linestyle='--')
        axs[i].fill_between(np.arange(len(mean_STABL)), np.log10(low_STABL), np.log10(high_STABL), color=rgblist[1],
                         alpha=0.1)

        # Plot PPO
        axs[i].plot(range(len(mean_PPO)), np.log10(mean_PPO), color=rgblist[2], label='PPO')
        axs[i].plot(range(len(mean_PPO)), np.log10(median_PPO), color=rgblist[2], linestyle='--')
        axs[i].fill_between(np.arange(len(mean_PPO)), np.log10(low_PPO), np.log10(high_PPO), color=rgblist[2], alpha=0.1)

        # Plot OPTIMAL
        axs[i].plot(range(len(mean_OPTIMAL)), np.log10(mean_OPTIMAL), color=rgblist[3], label='LQR')
        axs[i].plot(range(len(mean_OPTIMAL)), np.log10(median_OPTIMAL), color=rgblist[3], linestyle='--')
        axs[i].fill_between(np.arange(len(mean_OPTIMAL)), np.log10(low_OPTIMAL), np.log10(high_OPTIMAL), color=rgblist[3],
                         alpha=0.1)

        # Add labels, grid, and title for each subplot
        axs[i].grid(True)
        axs[i].set_ylabel("Logarithmic Cost $log_{10} c(t)$")
        axs[i].set_xlabel("Time step $t$")
        if noise_level == "noise01":
            s = "Scaling Factor $0.1$"
        elif noise_level == "standard":
            s = "Scaling Factor $1.0$"
        elif noise_level == "noise10":
            s = "Scaling Factor $10.0$"
        else:
            print("Unknown noise type: ".format(noise_level))
            raise Exception
        axs[i].set_title(f"{s}", fontsize=14)

        # Ensure the zero line is explicitly added and fixed
        axs[i].axhline(0, color='black', linestyle='--', linewidth=0.8)

    # Add legend to the last subplot
    axs[-1].legend(loc="best", fontsize=10)
    plt.tight_layout()

    # Ensure the directory exists for saving the plots
    plot_dir = 'plots/cost_of_t_triples/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Save the plot
    fig.savefig(f'{plot_dir}/cost_of_t_triples_{problem_name}.pdf', bbox_inches='tight')
    plt.close()


def plot_cum_regret_triplets(problem_name):
    noise_levels = ["noise01", "standard", "noise10"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Initialize variables to track global min and max for y-limits
    global_ymin = float('inf')
    global_ymax = float('-inf')

    # First loop: determine the global y-limits
    for i, noise_level in enumerate(noise_levels):
        # Load the cost data for the algorithms
        cost_ARBMLE = np.load(f'costs_of_t/Cost_ARBMLE_{noise_level}_{problem_name}.npy')
        cost_STABL = np.load(f'costs_of_t/Cost_STABL_{noise_level}_{problem_name}.npy')
        cost_PPO = np.load(f'costs_of_t/Cost_PPO_{noise_level}_{problem_name}.npy')
        cost_OPTIMAL = np.load(f'costs_of_t/Cost_OPTIMAL_{noise_level}_{problem_name}.npy')

        # Truncate to the shortest array length to avoid mismatches
        min_length = min(cost_ARBMLE.shape[1], cost_STABL.shape[1], cost_PPO.shape[1], cost_OPTIMAL.shape[1])
        cost_ARBMLE = cost_ARBMLE[:, :min_length]
        cost_STABL = cost_STABL[:, :min_length]
        cost_PPO = cost_PPO[:, :min_length]
        cost_OPTIMAL = cost_OPTIMAL[:, :min_length]

        # Compute cumulative regret
        cum_regret_ARBMLE = np.cumsum(cost_ARBMLE - cost_OPTIMAL, axis=1)
        cum_regret_STABL = np.cumsum(cost_STABL - cost_OPTIMAL, axis=1)
        cum_regret_PPO = np.cumsum(cost_PPO - cost_OPTIMAL, axis=1)

        # Get the mean, lower, and upper bounds for cumulative regret of each algorithm
        mean_ARBMLE, median_ARBMLE, low_ARBMLE, high_ARBMLE = get_percentile_confidence(cum_regret_ARBMLE)
        mean_STABL, median_STABL, low_STABL, high_STABL = get_percentile_confidence(cum_regret_STABL)
        mean_PPO, median_PPO, low_PPO, high_PPO = get_percentile_confidence(cum_regret_PPO)

        # Update global min and max for y-limits
        global_ymin = min(global_ymin, np.min(low_ARBMLE), np.min(low_STABL), np.min(low_PPO))
        global_ymax = max(global_ymax, np.max(high_ARBMLE), np.max(high_STABL), np.max(high_PPO), np.max(mean_ARBMLE),
                          np.max(mean_STABL), np.max(mean_PPO))

    # Second loop: plot with unified y-limits
    for i, noise_level in enumerate(noise_levels):
        # Load the cost data again
        cost_ARBMLE = np.load(f'costs_of_t/Cost_ARBMLE_{noise_level}_{problem_name}.npy')
        cost_STABL = np.load(f'costs_of_t/Cost_STABL_{noise_level}_{problem_name}.npy')
        cost_PPO = np.load(f'costs_of_t/Cost_PPO_{noise_level}_{problem_name}.npy')
        cost_OPTIMAL = np.load(f'costs_of_t/Cost_OPTIMAL_{noise_level}_{problem_name}.npy')

        # Truncate to the shortest array length
        min_length = min(cost_ARBMLE.shape[1], cost_STABL.shape[1], cost_PPO.shape[1], cost_OPTIMAL.shape[1])
        cost_ARBMLE = cost_ARBMLE[:, :min_length]
        cost_STABL = cost_STABL[:, :min_length]
        cost_PPO = cost_PPO[:, :min_length]
        cost_OPTIMAL = cost_OPTIMAL[:, :min_length]

        # Compute cumulative regret
        cum_regret_ARBMLE = np.cumsum(cost_ARBMLE - cost_OPTIMAL, axis=1)
        cum_regret_STABL = np.cumsum(cost_STABL - cost_OPTIMAL, axis=1)
        cum_regret_PPO = np.cumsum(cost_PPO - cost_OPTIMAL, axis=1)

        # Get the mean, lower, and upper bounds for cumulative regret of each algorithm
        mean_ARBMLE, median_ARBMLE, low_ARBMLE, high_ARBMLE = get_percentile_confidence(cum_regret_ARBMLE)
        mean_STABL, median_STABL, low_STABL, high_STABL = get_percentile_confidence(cum_regret_STABL)
        mean_PPO, median_PPO, low_PPO, high_PPO = get_percentile_confidence(cum_regret_PPO)

        # Plot ARBMLE
        axs[i].plot(range(len(mean_ARBMLE)), mean_ARBMLE, color=rgblist[0], label='ARBMLE')
        axs[i].plot(range(len(mean_ARBMLE)), median_ARBMLE, color=rgblist[0], linestyle='--')
        axs[i].fill_between(np.arange(len(mean_ARBMLE)), low_ARBMLE, high_ARBMLE, color=rgblist[0], alpha=0.1)

        # Plot STABL
        axs[i].plot(range(len(mean_STABL)), mean_STABL, color=rgblist[1], label='STABL')
        axs[i].plot(range(len(mean_STABL)), median_STABL, color=rgblist[1], linestyle='--')
        axs[i].fill_between(np.arange(len(mean_STABL)), low_STABL, high_STABL, color=rgblist[1], alpha=0.1)

        # Plot PPO
        axs[i].plot(range(len(mean_PPO)), mean_PPO, color=rgblist[2], label='PPO')
        axs[i].plot(range(len(mean_PPO)), median_PPO, color=rgblist[2], linestyle='--')
        axs[i].fill_between(np.arange(len(mean_PPO)), low_PPO, high_PPO, color=rgblist[2], alpha=0.1)

        # Set y-axis to symlog scale
        axs[i].set_yscale('symlog', linthresh=1)

        # Set unified y-limits
        axs[i].set_ylim(global_ymin*2, global_ymax*2)

        # Add grid, labels, and titles
        axs[i].grid(True)
        axs[i].set_ylabel("Cumulative Regret")
        axs[i].set_xlabel("Time step $t$")
        if noise_level == "noise01":
            s = "Scaling Factor $0.1$"
        elif noise_level == "standard":
            s = "Scaling Factor $1.0$"
        elif noise_level == "noise10":
            s = "Scaling Factor $10.0$"
        else:
            print("Unknown noise type: ".format(noise_level))
            raise Exception
        axs[i].set_title(f"{s}", fontsize=14)

    # Add legend to the last subplot
    axs[-1].legend(loc="best", fontsize=10)

    # Align zero across the subplots
    for ax in axs:
        ax.axhline(0, color='black', linestyle='--', linewidth=0.8)

    # Tight layout
    plt.tight_layout()

    # Ensure the directory exists for saving the plots
    plot_dir = 'plots/cum_regret_triples/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Save the plot
    fig.savefig(f'{plot_dir}/cum_regret_triples_{problem_name}.pdf', bbox_inches='tight')
    plt.close()


def plot_cum_regret(save_file_name, problem_name):
    # Load the cost data for the algorithms and LQR controller
    cost_ARBMLE = np.load(f'costs_of_t/Cost_ARBMLE_{save_file_name}_{problem_name}.npy')
    cost_STABL = np.load(f'costs_of_t/Cost_STABL_{save_file_name}_{problem_name}.npy')
    cost_PPO = np.load(f'costs_of_t/Cost_PPO_{save_file_name}_{problem_name}.npy')
    cost_OPTIMAL = np.load(f'costs_of_t/Cost_OPTIMAL_{save_file_name}_{problem_name}.npy')

    # Ensure all cost arrays have the same length by truncating to the shortest array length
    min_length_ARBMLE = min(cost_ARBMLE.shape[1], cost_OPTIMAL.shape[1])
    min_length_STABL = min(cost_STABL.shape[1], cost_OPTIMAL.shape[1])
    min_length_PPO = min(cost_PPO.shape[1], cost_OPTIMAL.shape[1])

    tot_min = min(min_length_PPO, min_length_ARBMLE, min_length_STABL)
    print(f"{save_file_name}_{problem_name}: min = {tot_min}\n")

    cost_ARBMLE = cost_ARBMLE[:, :min_length_ARBMLE]
    cost_STABL = cost_STABL[:, :min_length_STABL]
    cost_PPO = cost_PPO[:, :min_length_PPO]
    cost_OPTIMAL = cost_OPTIMAL[:, :min(min_length_ARBMLE, min_length_STABL, min_length_PPO)]

    # Compute regret as the difference between each algorithm's cost and the cost of the optimal controller at time step t
    regret_ARBMLE = cost_ARBMLE - cost_OPTIMAL
    regret_STABL = cost_STABL - cost_OPTIMAL
    regret_PPO = cost_PPO - cost_OPTIMAL

    # Compute cumulative regret
    cum_regret_ARBMLE = np.cumsum(regret_ARBMLE, axis=1)
    cum_regret_STABL = np.cumsum(regret_STABL, axis=1)
    cum_regret_PPO = np.cumsum(regret_PPO, axis=1)

    # Get the mean, lower, and upper bounds for cumulative regret of each algorithm
    mean_ARBMLE, median_ARBMLE, low_ARBMLE, high_ARBMLE = get_percentile_confidence(cum_regret_ARBMLE)
    mean_STABL, median_STABL, low_STABL, high_STABL = get_percentile_confidence(cum_regret_STABL)
    mean_PPO, median_PPO, low_PPO, high_PPO = get_percentile_confidence(cum_regret_PPO)

    # Create the plot
    fig = plt.figure()

    # Use a symlog scale for the y-axis (logarithmic for large values, linear for small values)
    linthresh = 1  # The threshold below which it switches to a linear scale

    # Plot cumulative regret for ARBMLE
    plt.plot(range(len(mean_ARBMLE)), mean_ARBMLE, color=rgblist[0], label='ARBMLE')
    plt.plot(range(len(mean_ARBMLE)), median_ARBMLE, color=rgblist[0], linestyle='--')
    plt.fill_between(np.arange(len(mean_ARBMLE)), low_ARBMLE, high_ARBMLE, color=rgblist[0], alpha=0.1)

    # Plot cumulative regret for STABL
    plt.plot(range(len(mean_STABL)), mean_STABL, color=rgblist[1], label='STABL')
    plt.plot(range(len(mean_STABL)), median_STABL, color=rgblist[1], linestyle='--')
    plt.fill_between(np.arange(len(mean_STABL)), low_STABL, high_STABL, color=rgblist[1], alpha=0.1)

    # Plot cumulative regret for PPO
    plt.plot(range(len(mean_PPO)), mean_PPO, color=rgblist[2], label='PPO')
    plt.plot(range(len(mean_PPO)), median_PPO, color=rgblist[2], linestyle='--')
    plt.fill_between(np.arange(len(mean_PPO)), low_PPO, high_PPO, color=rgblist[2], alpha=0.1)

    # Set y-axis to symlog scale
    plt.yscale('symlog', linthresh=linthresh)

    # Add labels and title
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True)
    plt.ylabel('Cumulative Regret', fontsize=16)
    plt.xlabel('Time step $t$', fontsize=16)

    # Ensure the directory exists for saving the plots
    plot_dir = 'plots/cum_regret/'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Save the plot
    fig.savefig(f'{plot_dir}/cum_regret_{save_file_name}_{problem_name}.pdf', bbox_inches='tight')
    plt.close()


def main(
    run_optimal_tests=True,
    run_tests=True,
    run_single_case=False,
    convert_cum_cost_to_cost=True,
    bool_plot_cum_cost=True,
    bool_plot_cost_of_t=True,
    bool_plot_cum_regret=True,
    bool_plot_cost_progression=True,
    bool_plot_triplets=False):
    """
    Main function to execute experiments and generate plots based on specified flags.

    Parameters:
        run_optimal_tests (bool): Run experiments using the optimal controller if True.
        run_tests (bool): Run all experiments with adaptive algorithms if True.
        run_single_case (bool): Run a single experiment instance if True.
        convert_cum_cost_to_cost (bool): Convert cumulative costs to per-step costs if True.
        bool_plot_cum_cost (bool): Generate plots of cumulative costs if True.
        bool_plot_cost_of_t (bool): Generate plots of cost over time if True.
        bool_plot_cum_regret (bool): Generate plots of cumulative regret if True.
        bool_plot_cost_progression (bool): Plot cost progression for individual runs if True.
        bool_plot_triplets (bool): Generate triplet plots for cost and cumulative regret if True.

    This function orchestrates the execution of experiments and the generation of plots
    based on the provided boolean flags. It can run optimal controller tests, adaptive
    algorithm tests, convert cost data, and produce various plots for analysis.
    """

    # to run optimal controller for all tests:
    if run_optimal_tests:
        run_all_tests(run_optimal=True)

    # to run all tests:
    if run_tests:
        run_all_tests(run_optimal=False)

    # to run single instance
    if run_single_case:
        run_single()

    # to convert cumulative cost to cost(t)
    if convert_cum_cost_to_cost:
        algorithms = ["ARBMLE", "STABL", "PPO", "OPTIMAL"]
        save_names = ["standard", "noise01", "noise10", "noise1skewPlu10", "noise1skewMin10", "noise10skewPlu10",
                      "noise10skewMin10", "disturbance", "standard_longrun", "nonlinear_x_squared",
                      "nonlinear_x_sin(x)", "nonlinear_x_log(x)"]
        problem_names = ["Boeing", "UAV", "CartPole"]
        param_grid = list(itertools.product(algorithms, save_names, problem_names))

        for (algorithm, save_file_name, problem_name) in param_grid:
            filename = f"{algorithm}_{save_file_name}_{problem_name}"
            cumulative_cost_to_cost(filename)

    # to plot the cumulative cost (instead of the cumulative regret). Only required for standard_Boeing and -UAV
    if bool_plot_cum_cost:
        save_names = ["standard"]
        problem_names = ["Boeing", "UAV"]
        param_grid = list(itertools.product(save_names, problem_names))

        for (save_file_name, problem_name) in param_grid:
            generate_cum_cost_plot(save_file_name, problem_name)

    # to plot cost(t)
    if bool_plot_cost_of_t:
        save_names = ["standard", "noise01", "noise10", "noise1skewPlu10", "noise1skewMin10", "noise10skewPlu10",
                      "noise10skewMin10", "disturbance", "standard_longrun"]
        problem_names = ["Boeing", "UAV", "CartPole"]
        param_grid = list(itertools.product(save_names, problem_names))

        for (save_file_name, problem_name) in param_grid:
            plot_cost_of_t(save_file_name, problem_name)

    # to plot cumulative regret
    if bool_plot_cum_regret:
        save_names = \
            ["standard", "noise01", "noise10", "noise1skewPlu10", "noise1skewMin10", "noise10skewPlu10",
             "noise10skewMin10", "disturbance", "standard_longrun"]
        problem_names = ["Boeing", "UAV", "CartPole"]
        param_grid = list(itertools.product(save_names, problem_names))

        for (save_file_name, problem_name) in param_grid:
            plot_cum_regret(save_file_name, problem_name)

    # to plot progressive cost(t) (only for graphs showing cost development in nonlinear case)
    if bool_plot_cost_progression:
        save_names = ["nonlinear_x_log(x)", "nonlinear_x_squared", "nonlinear_x_sin(x)"]
        problem_names = ["Boeing", "UAV", "CartPole"]
        param_grid = list(itertools.product(save_names, problem_names))

        for (save_file_name, problem_name) in param_grid:
            plot_cost_progression_for_runs(save_file_name, problem_name)

    # to plot cost(t) and cumulative regret as triplets
    if bool_plot_triplets:
        problem_names = ["Boeing", "UAV", "CartPole"]
        for problem_name in problem_names:
            plot_cost_of_t_triple(problem_name)
            plot_cum_regret_triplets(problem_name)


if __name__ == '__main__':
    main()



