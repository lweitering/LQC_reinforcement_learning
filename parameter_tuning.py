"""
This file is used to tune the hyperparameters of the different algorithms. Each parameter constellation is run
using a GridSearch-approach. Results are stored in JSON-format for later evaluation. The methods for the evaluation
can also be found in this file. Evaluation will be done using a ranking system.

@author: Lucas Weitering
"""

import itertools
import os

import numpy as np
import json
import time
import pandas as pd
from colorama import Back

import dynamics
from algorithms.adaptive_controller import ProximalPolicyOptimization, AdaptiveController


def get_problem_formulations():
    return {
        'Boeing': dynamics.boeing(),
        'UAV': dynamics.uav(),
        'CartPole': dynamics.cart_pole(),
    }


def run_ppo_experiment(A_star, B_star, Q, R, horizon, T_init, reg, cur_seed, gamma, lambda_, epsilon, lr_policy,
                       lr_value, std_clamp_max=0.5, update_steps=10, problem=None):
    """
    Run a PPO experiment with specified hyperparameters and return the cumulative cost.
    """
    # Define the experiment with the given hyperparameters
    ppo = ProximalPolicyOptimization(A_star, B_star, Q, R, horizon, T_init, reg, 'PPO', cur_seed=cur_seed,
                                     gamma=gamma, lambda_=lambda_, epsilon=epsilon, lr_policy=lr_policy,
                                     lr_value=lr_value, std_clamp_max=std_clamp_max, update_steps=update_steps)

    # Generate random noise for the experiment
    input_noise = np.random.normal(0, size=(B_star.shape[1], T_init))
    if problem == "CartPole":
        input_noise *= 10
    noise = np.random.normal(0, size=(A_star.shape[0], horizon + T_init))

    # Run the experiment and return the cumulative cost
    ppo.run_experiment(noise, input_noise, reg)
    return np.sum(ppo.costs[T_init:])


def grid_search_ppo_multiple_problems(horizon, T_init, reg, num_repeats, output_file):
    """
    Perform a grid search over Proximal Policy Optimization hyperparameters across multiple problems.

    This function tests various combinations of PPO hyperparameters (gamma, lambda, epsilon,
    learning rates, standard deviation clamp, update steps) on multiple control problems.
    It runs each combination for a specified number of repeats, averages the costs,
    and saves the results to an output file. Previously tested combinations are skipped
    to avoid redundant computations.

    Parameters:
        horizon (int): Length of each experiment run.
        T_init (int): Initial time steps using a stabilizing controller.
        reg (float): Regularization parameter.
        num_repeats (int): Number of repetitions for averaging.
        output_file (str): File path to save the results.
    """
    # Define the hyperparameter grid
    gamma_values = [1.0]  # Discount factor for future rewards
    lambda_values = [1.0]  # Generalised Advantage Estimation discount factor
    epsilon_values = [0.1, 0.2, 0.3]  # Clipping parameter for PPO loss
    lr_policy_values = [1e-3, 2.5e-3, 7.5e-3, 1e-4]  # Learning rate for the policy network
    lr_value_values = [1e-4, 5e-4, 7.5e-3, 1e-3, 5e-2]  # Learning rate for the value network
    std_clamp_max_values = [0.1, 0.3, 0.5, 0.7]  # Clamping on the standard deviation of the policy
    update_steps_values = [5, 10, 20]  # Number of steps before PPO is updated (=where the number is missing, assume 10)

    # Get the problem formulations
    problems = get_problem_formulations()

    # Create the Cartesian product of all hyperparameters
    param_grid = list(itertools.product(gamma_values, lambda_values, epsilon_values,
                                        lr_policy_values, lr_value_values, std_clamp_max_values, update_steps_values))

    # Read existing results to avoid redundant computations
    tested_combinations = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Extract the parameter combination as a tuple and add to the set
                param_tuple = (data['gamma'], data['lambda'], data['epsilon'],
                               data['lr_policy'], data['lr_value'], data['std_clamp_max'], data.get('update_steps', 10))
                tested_combinations.add(param_tuple)

    # Filter out already tested combinations
    param_grid = [params for params in param_grid if params not in tested_combinations]

    # Measure the time taken for a single parameter set
    total_combinations = len(param_grid)
    start_time = time.time()

    # Open the output file to save results
    with open(output_file, 'a') as f:
        # Loop over each combination of hyperparameters
        for index, param_combination in enumerate(param_grid):
            gamma, lambda_, epsilon, lr_policy, lr_value, std_clamp_max, update_steps = param_combination
            results_for_problems = {}
            failed_experiment = False  # Track if the experiment fails

            print(f"Experiment {index + 1} / {len(param_grid)}")
            print(f"Running PPO with parameters: gamma={gamma}, lambda={lambda_}, epsilon={epsilon}, "
                  f"lr_policy={lr_policy}, lr_value={lr_value}, std_clamp_max={std_clamp_max}, "
                  f"update_steps={update_steps}")

            try:
                # Loop over all problems and store results separately
                for problem_name, (A_star, B_star, Q, R, _) in problems.items():
                    problem_costs = []

                    print(f"Evaluating on problem: {problem_name}")
                    for i in range(num_repeats):
                        np.random.seed(i)
                        avg_cost = run_ppo_experiment(A_star, B_star, Q, R, horizon, T_init, reg, i, gamma, lambda_,
                                                      epsilon, lr_policy, lr_value, std_clamp_max, update_steps,
                                                      problem=problem_name)
                        problem_costs.append(avg_cost)

                    # Calculate the average cost for this problem
                    avg_cost_per_problem = np.mean(problem_costs)
                    results_for_problems[problem_name] = avg_cost_per_problem

            except Exception as e:
                print(f"Experiment failed for parameters: gamma={gamma}, lambda={lambda_}, epsilon={epsilon}, "
                      f"lr_policy={lr_policy}, lr_value={lr_value}, std_clamp_max={std_clamp_max}, "
                      f"update_steps={update_steps} on problem {problem_name}. Error: {str(e)}")
                failed_experiment = True

            # Save the hyperparameters and results or failure status to the file
            result = {
                'gamma': gamma,
                'lambda': lambda_,
                'epsilon': epsilon,
                'lr_policy': lr_policy,
                'lr_value': lr_value,
                'std_clamp_max': std_clamp_max,
                'update_steps': update_steps,
                'costs': results_for_problems if not failed_experiment else {},
                'failed': failed_experiment  # Mark failed experiments
            }

            # Write result to file (appending after each run to ensure progress is saved)
            f.write(json.dumps(result) + '\n')
            f.flush()  # Ensure that the data is written to the file immediately

            if failed_experiment:
                print(f"Marked as failed: {result} - Experiment {index + 1}/{len(param_grid)}")

            # If this is the first parameter combination, calculate the average time per set
            if index == 0:
                end_time = time.time()
                time_for_one_combination = end_time - start_time
                estimated_total_time = time_for_one_combination * total_combinations

                # Print the time required for one parameter constellation and estimate total time
                print(Back.GREEN + f"Time for one parameter constellation: {time_for_one_combination:.2f} seconds")
                print(Back.GREEN + f"Estimated total time for {total_combinations} parameter constellations: "
                                   f"{estimated_total_time / 60:.2f} minutes ({estimated_total_time / 3600:.2f} hours)"
                      + Back.RESET)


def analyze_parameter_tuning_results(input_file, algorithm, output_results_file):
    """
    Analyze parameter tuning results to find the best parameter configuration.

    Parameters:
        input_file (str): Path to the JSON file containing tuning results.
        algorithm (str): Name of the algorithm ('PPO', 'ARBMLE', or 'STABL').
        output_results_file (str): Path to save the analysis results.

    The function reads tuning results, filters out failed and duplicate entries,
    ranks parameter configurations based on cost across problems, identifies the
    best parameters, and writes the results to an output file.

    Returns:
        dict: Best parameter configuration.
    """

    # Load the data from the JSON file
    parameter_data = []
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            if not data['failed']:  # Only keep entries where 'failed' is False
                parameter_data.append(data)

    if not parameter_data:
        print("No valid parameter constellations found.")
        return

    # Remove duplicates based on parameter values
    unique_entries = []
    seen_configs = set()

    for entry in parameter_data:
        # Define the parameter set as a tuple to check uniqueness
        if algorithm == 'PPO':
            config_tuple = (
                entry['gamma'], entry['lambda'], entry['epsilon'], entry['lr_policy'],
                entry['lr_value'], entry['std_clamp_max'], entry.get('update_steps', 10)
            )
        elif algorithm in ['ARBMLE', 'STABL']:
            config_tuple = (
                entry['num_restarts'], entry['max_iters'], entry['step_size'], entry['rel_tol'],
                entry['delta'], entry['bias'], entry.get('T_w', None), entry.get('sigma_w', None),
                entry['S']
            )

        if config_tuple not in seen_configs:
            seen_configs.add(config_tuple)
            unique_entries.append(entry)

    parameter_data = unique_entries

    print(f"Number of valid constellations: {len(parameter_data)}")

    # Extract the costs per problem and corresponding parameters
    costs_per_problem = {}
    param_details = []

    for entry in parameter_data:
        if algorithm == 'PPO':
            param_details.append({
                'gamma': entry['gamma'],
                'lambda': entry['lambda'],
                'epsilon': entry['epsilon'],
                'lr_policy': entry['lr_policy'],
                'lr_value': entry['lr_value'],
                'std_clamp_max': entry['std_clamp_max'],
                'update_steps': entry.get('update_steps', 10)
            })
        elif algorithm == 'ARBMLE' or algorithm == 'STABL':
            param_details.append({
                'num_restarts': entry['num_restarts'],
                'max_iters': entry['max_iters'],
                'step_size': entry['step_size'],
                'rel_tol': entry['rel_tol'],
                'delta': entry['delta'],
                'bias': entry['bias'],
                'T_w': entry.get('T_w', None),
                'sigma_w': entry.get('sigma_w', None),
                'S': entry['S']
            })

        for problem, cost in entry['costs'].items():
            if problem not in costs_per_problem:
                costs_per_problem[problem] = []
            costs_per_problem[problem].append(cost)

    # Create a DataFrame for the cost data for easier processing
    cost_df = pd.DataFrame(costs_per_problem)

    # Rank the constellations for each problem (lower cost = better rank)
    ranks = cost_df.rank(method='min', axis=0)

    # Sum the ranks for each parameter constellation
    ranks['total_rank'] = ranks.sum(axis=1)

    # Find the parameter constellation with the lowest sum of ranks
    best_index = ranks['total_rank'].idxmin()
    best_params = param_details[best_index]
    best_costs = cost_df.iloc[best_index].to_dict()

    # Prepare the results to print and save to file
    results_lines = []
    if algorithm == 'PPO':
        results_lines.append("Best PPO parameter constellation:")
        results_lines.append(f"  Gamma: {best_params['gamma']}")
        results_lines.append(f"  Lambda: {best_params['lambda']}")
        results_lines.append(f"  Epsilon: {best_params['epsilon']}")
        results_lines.append(f"  Learning Rate (Policy): {best_params['lr_policy']}")
        results_lines.append(f"  Learning Rate (Value): {best_params['lr_value']}")
        results_lines.append(f"  Std Clamp Max: {best_params['std_clamp_max']}")
        results_lines.append(f"  Update Steps: {best_params['update_steps']}")
    elif algorithm == 'ARBMLE':
        results_lines.append("Best ARBMLE parameter constellation:")
        results_lines.append(f"  Num Restarts: {best_params['num_restarts']}")
        results_lines.append(f"  Max Iters: {best_params['max_iters']}")
        results_lines.append(f"  Step Size: {best_params['step_size']}")
        results_lines.append(f"  Rel Tol: {best_params['rel_tol']}")
        results_lines.append(f"  Delta: {best_params['delta']}")
        results_lines.append(f"  Bias: {best_params['bias']}")
        results_lines.append(f"  S: {best_params['S']}")
    elif algorithm == 'STABL':
        results_lines.append("Best STABL parameter constellation:")
        results_lines.append(f"  Num Restarts: {best_params['num_restarts']}")
        results_lines.append(f"  Max Iters: {best_params['max_iters']}")
        results_lines.append(f"  Step Size: {best_params['step_size']}")
        results_lines.append(f"  Rel Tol: {best_params['rel_tol']}")
        results_lines.append(f"  Delta: {best_params['delta']}")
        results_lines.append(f"  Bias: {best_params['bias']}")
        results_lines.append(f"  T_w: {best_params['T_w']}")
        results_lines.append(f"  Sigma_w: {best_params['sigma_w']}")
        results_lines.append(f"  S: {best_params['S']}")

    results_lines.append("\nCosts for each problem:")
    for problem, cost in best_costs.items():
        results_lines.append(f"  {problem}: {cost}")

    # Print results to the console
    for line in results_lines:
        print(line)

    # Write results to the output file if provided
    if output_results_file:
        with open(output_results_file, 'w') as f_out:
            for line in results_lines:
                f_out.write(line + '\n')

    return best_params


# Methods for parameter tuning of ARBMLE and StabL
def run_model_based_experiment(A_star, B_star, Q, R, horizon, T_init, reg, cur_seed, num_restarts, max_iters, step_size,
                               rel_tol, delta, bias, S, algorithm, T_w=35, sigma_w=2, problem=None):
    """
    Run a model-based adaptive control experiment and return the cumulative cost.

    Parameters:
        A_star, B_star, Q, R: System matrices defining the dynamics and cost.
        horizon (int): Total time horizon for the experiment.
        T_init (int): Initial time steps using a stabilizing controller.
        reg (float): Regularization parameter.
        cur_seed (int): Random seed for reproducibility.
        num_restarts, max_iters, step_size, rel_tol, delta, bias, S: Algorithm hyperparameters.
        algorithm (str): Algorithm name ('ARBMLE' or 'STABL').
        T_w (int, optional): Time window parameter (used in 'STABL').
        sigma_w (float, optional): Noise standard deviation parameter.
        problem (str, optional): Problem name for specific adjustments.

    Returns:
        float: Cumulative cost after T_init steps.
    """
    # Define the experiment with the given hyperparameters
    alg = AdaptiveController(A_star, B_star, Q, R, horizon, T_init, reg, algorithm, cur_seed=cur_seed,
                             num_restarts=num_restarts, max_iters=max_iters, step_size=step_size,
                             rel_tol=rel_tol, delta=delta, bias=bias, T_w=T_w, sigma_w=sigma_w, S=S)

    # Generate random noise for the experiment
    input_noise = np.random.normal(0, size=(B_star.shape[1], T_init))
    if problem == "CartPole":
        input_noise *= 10
    noise = np.random.normal(0, size=(A_star.shape[0], horizon + T_init))

    # Run the experiment and return the cumulative cost
    alg.run_experiment(noise, input_noise, reg)
    return np.sum(alg.costs[T_init:])


def grid_search_model_based_multiple_problems(horizon, T_init, reg, num_repeats, output_file, algorithm):
    """
    Perform grid search over hyperparameters for ARBMLE or STABL across multiple problems.

    Parameters:
        horizon (int): Experiment length.
        T_init (int): Initial stabilizing steps.
        reg (float): Regularization parameter.
        num_repeats (int): Number of repetitions for averaging.
        output_file (str): File to save results.
        algorithm (str): 'ARBMLE' or 'STABL'.

    This function tests various hyperparameter combinations for the specified algorithm
    on multiple control problems, records the average costs, and saves the results.
    """
    # Define the hyperparameter grid
    num_restarts_values, max_iters_values, step_size_values, rel_tol_values, delta_values, bias_values, \
        T_w_values, sigma_w_values, S_values = None, None, None, None, None, None, None, None, None

    if algorithm == 'ARBMLE':
        num_restarts_values = [5]  # Number of restarts for optimization
        max_iters_values = [500]  # Maximum iterations for optimization
        step_size_values = [0.1, 0.05, 0.01]  # Step size for the gradient descent
        rel_tol_values = [1e-4]  # Relative tolerance for convergence
        delta_values = [1e-2, 1e-3, 1e-4]  # Delta parameter for robustness
        bias_values = [0.1, 0.05, 0.01, 0.005]  # Bias in the estimation process
        S_values = [20, 50, 80, 100]  # Scaling term - bound on parameter space
    elif algorithm == 'STABL':
        num_restarts_values = [5]  # Number of restarts for optimization
        max_iters_values = [500]  # Maximum iterations for optimization
        step_size_values = [0.1, 0.05, 0.01]  # Step size for the gradient descent
        rel_tol_values = [1e-4]  # Relative tolerance for convergence
        delta_values = [1e-2, 1e-3, 1e-4]  # Delta parameter for robustness
        bias_values = [0.01]  # Bias in the estimation process
        T_w_values = [10, 30, 50]  # Window length for exploration-exploitation balance
        sigma_w_values = [0.5, 1, 2]  # Standard deviation of the noise in the dynamics
        S_values = [20, 50]  # Scaling term - bound on parameter space
    else:
        print("Unknown Algorithm - Returning")
        return

    problems = get_problem_formulations()

    # Create the Cartesian product of all hyperparameters
    param_grid = []
    if algorithm == "ARBMLE":
        param_grid = list(itertools.product(num_restarts_values, max_iters_values, step_size_values,
                                            rel_tol_values, delta_values, bias_values, S_values))
    elif algorithm == "STABL":
        param_grid = list(itertools.product(num_restarts_values, max_iters_values, step_size_values,
                                            rel_tol_values, delta_values, bias_values,
                                            T_w_values, sigma_w_values, S_values))

    # Read existing results to avoid redundant computations
    tested_combinations = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Extract the parameter combination as a tuple and add to the set
                if algorithm == "ARBMLE":
                    param_tuple = (data['num_restarts'], data['max_iters'], data['step_size'], data['rel_tol'],
                                   data['delta'], data['bias'], data['S'])
                elif algorithm == "STABL":
                    param_tuple = (data['num_restarts'], data['max_iters'], data['step_size'], data['rel_tol'],
                                   data['delta'], data['bias'], data['T_w'], data['sigma_w'], data['S'])
                tested_combinations.add(param_tuple)

    # Filter out already tested combinations
    param_grid = [params for params in param_grid if params not in tested_combinations]

    # Measure the time taken for a single parameter set
    total_combinations = len(param_grid)
    start_time = time.time()

    # Open the output file to save results
    with open(output_file, 'a') as f:
        # Loop over each combination of hyperparameters
        for index, param_combination in enumerate(param_grid):
            if algorithm == "ARBMLE":
                num_restarts, max_iters, step_size, rel_tol, delta, bias, S = param_combination
            elif algorithm == "STABL":
                num_restarts, max_iters, step_size, rel_tol, delta, bias, T_w, sigma_w, S = param_combination
            results_for_problems = {}
            failed_experiment = False  # Track if the experiment fails

            print(f"Experiment {index + 1} / {len(param_grid)}")
            if algorithm == "ARBMLE":
                print(f"Running {algorithm} with parameters: num_restarts={num_restarts}, max_iters={max_iters}, "
                      f"step_size={step_size}, rel_tol={rel_tol}, delta={delta}, bias={bias}, S={S}")
            elif algorithm == "STABL":
                print(f"Running {algorithm} with parameters: num_restarts={num_restarts}, max_iters={max_iters}, "
                      f"step_size={step_size}, rel_tol={rel_tol}, delta={delta}, bias={bias}, "
                      f"T_w={T_w}, sigma_w={sigma_w}, S={S}")

            try:
                # Loop over all problems and store results separately
                for problem_name, (A_star, B_star, Q, R, _) in problems.items():
                    problem_costs = []

                    print(f"Evaluating on problem: {problem_name}")
                    for i in range(num_repeats):
                        np.random.seed(i)
                        if algorithm == "ARBMLE":
                            avg_cost = run_model_based_experiment(A_star=A_star, B_star=B_star, Q=Q, R=R,
                                                                  horizon=horizon, T_init=T_init, reg=reg, cur_seed=i,
                                                                  num_restarts=num_restarts, max_iters=max_iters,
                                                                  step_size=step_size, rel_tol=rel_tol, delta=delta,
                                                                  bias=bias, S=S, algorithm=algorithm,
                                                                  problem=problem_name)
                        elif algorithm == "STABL":
                            avg_cost = run_model_based_experiment(A_star=A_star, B_star=B_star, Q=Q, R=R,
                                                                  horizon=horizon, T_init=T_init, reg=reg, cur_seed=i,
                                                                  num_restarts=num_restarts, max_iters=max_iters,
                                                                  step_size=step_size, rel_tol=rel_tol, delta=delta,
                                                                  bias=bias, S=S, algorithm=algorithm, T_w=T_w,
                                                                  sigma_w=sigma_w, problem=problem_name)
                        problem_costs.append(avg_cost)

                    # Calculate the average cost for this problem
                    avg_cost_per_problem = np.mean(problem_costs)
                    results_for_problems[problem_name] = avg_cost_per_problem

            except Exception as e:
                if algorithm == "ARBMLE":
                    print(f"Experiment failed for parameters: num_restarts={num_restarts}, max_iters={max_iters}, "
                          f"step_size={step_size}, rel_tol={rel_tol}, delta={delta}, bias={bias}, S={S} on problem "
                          f"{problem_name} for algorithm {algorithm}. Error: {str(e)}")
                elif algorithm == "STABL":
                    print(f"Experiment failed for parameters: num_restarts={num_restarts}, max_iters={max_iters}, "
                          f"step_size={step_size}, rel_tol={rel_tol}, delta={delta}, bias={bias}, "
                          f"T_w={T_w}, sigma_w={sigma_w}, S={S} on problem {problem_name} for algorithm {algorithm}. "
                          f"Error: {str(e)}")
                failed_experiment = True

            # Save the hyperparameters and results or failure status to the file
            if algorithm == "ARBMLE":
                result = {
                    'num_restarts': num_restarts,
                    'max_iters': max_iters,
                    'step_size': step_size,
                    'rel_tol': rel_tol,
                    'delta': delta,
                    'bias': bias,
                    'S': S,
                    'costs': results_for_problems if not failed_experiment else {},
                    'failed': failed_experiment  # Mark failed experiments
                }
            elif algorithm == "STABL":
                result = {
                    'num_restarts': num_restarts,
                    'max_iters': max_iters,
                    'step_size': step_size,
                    'rel_tol': rel_tol,
                    'delta': delta,
                    'bias': bias,
                    'T_w': T_w,
                    'sigma_w': sigma_w,
                    'S': S,
                    'costs': results_for_problems if not failed_experiment else {},
                    'failed': failed_experiment  # Mark failed experiments
                }

            # Write result to file (appending after each run to ensure progress is saved)
            f.write(json.dumps(result) + '\n')
            f.flush()  # Ensure that the data is written to the file immediately

            if failed_experiment:
                print(f"Marked as failed: {result} - Experiment {index + 1}/{len(param_grid)}")

            # If this is the first parameter combination, calculate the average time per set
            if index == 0:
                end_time = time.time()
                time_for_one_combination = end_time - start_time
                estimated_total_time = time_for_one_combination * total_combinations

                # Print the time required for one parameter constellation and estimate total time
                print(f"Time for one parameter constellation: {time_for_one_combination:.2f} seconds")
                print(f"Estimated total time for {total_combinations} parameter constellations: "
                      f"{estimated_total_time / 60:.2f} minutes ({estimated_total_time / 3600:.2f} hours)" + Back.RESET)

def main():
    # Hyperparameter tuning setup
    horizon = 200
    T_init = 30
    reg = 1e-4
    num_repeats = 20  # Number of repetitions for averaging results across each problem
    output_file_ppo = 'ppo_hyperparameter_tuning_results_multiple_problems.json'
    output_file_arbmle = 'arbmle_hyperparameter_tuning_results_multiple_problems.json'
    output_file_stabl = 'stabl_hyperparameter_tuning_results_multiple_problems.json'

    # Start the grid search for PPO across multiple problems
    grid_search_ppo_multiple_problems(horizon, T_init, reg, num_repeats, output_file_ppo)

    # Start the grid search for ARBMLE across multiple problems
    alg = "ARBMLE"
    grid_search_model_based_multiple_problems(horizon, T_init, reg, num_repeats, output_file_arbmle, alg)

    # Start the grid search for STABL across multiple problems
    alg = "STABL"
    grid_search_model_based_multiple_problems(horizon, T_init, reg, num_repeats, output_file_stabl, alg)


def analyse_results():
    result_file = "parameter_constellations.txt"
    output_file_ppo = 'ppo_hyperparameter_tuning_results_multiple_problems.json'
    output_file_arbmle = 'arbmle_hyperparameter_tuning_results_multiple_problems.json'
    output_file_stabl = 'stabl_hyperparameter_tuning_results_multiple_problems.json'

    analyze_parameter_tuning_results(output_file_ppo, "PPO", result_file)
    analyze_parameter_tuning_results(output_file_arbmle, "ARBMLE", result_file)
    analyze_parameter_tuning_results(output_file_stabl, "STABL", result_file)


if __name__ == '__main__':
    main()
    analyse_results()



