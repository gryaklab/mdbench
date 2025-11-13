
# Adapted from https://github.com/sdascoli/odeformer/tree/main

import re
import argparse
from pathlib import Path
from collections import namedtuple

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from tqdm import tqdm
import strogatz_ode

np.random.seed(0)

OdeDataset = namedtuple('OdeDataset', ['t', 'u', 'du'])

config = {
    "t_span": (0, 10),  # time span for integration
    "method": "LSODA",  # method for integration
    "rtol": 1e-5,  # relative tolerance (let's be strict)
    "atol": 1e-7,  # absolute tolerance (let's be strict)
    "first_step": 1e-6,  # initial step size (let's be strict)
    "t_eval": np.linspace(0, 10, 150),  # output times for the solution
    "min_step": 1e-10,  # minimum step size (only for LSODA)
}

matplotlib_rc = {
#'text': {'usetex': True},
'font': {'size': '16', 'family': 'serif'},#, 'serif': 'Palatino'},
'figure': {'titlesize': '20'},
'axes': {'titlesize': '22', 'labelsize': '28'},
'xtick': {'labelsize': '22'},
'ytick': {'labelsize': '22'},
'lines': {'linewidth': 3, 'markersize': 10},
'grid': {'color': 'grey', 'linestyle': 'solid', 'linewidth': 0.5},
}

def validate_equations(equations):
    """Validates the equations to make sure they are in the correct format.

    These are just a bunch of basic checks, which would probably all throw errors
    when trying to solve them anyway, but were useful to get the equations right
    in the beginning.
    """
    for eq_dict in equations:
        eq_string = eq_dict['eq']
        dim = eq_dict['dim']
        consts_values = eq_dict['consts']
        init_values = eq_dict['init']
        id = eq_dict['id']
        individual_eqs = eq_string.split('|')
        if len(individual_eqs) != dim:
            print(f"Error in equation {id}: The number of equations does not match the dimension.")

        highest_x_index = max([int(x[2:]) for x in re.findall(r'x_\d+', eq_string)])
        if highest_x_index + 1 != dim:
            pass #print(f"Warning in equation {id}: Found x_{highest_x_index} as highest index, but the dimension is {dim}.")

        const_indices = [int(c[2:]) for c in re.findall(r'c_\d+', eq_string)]
        if len(const_indices) > 0:
            highest_const_index = max(const_indices)
            for j in range(highest_const_index + 1):
                if f'c_{j}' not in eq_string:
                    print(f"Warning in equation {id}: c_{j} not appearing even though c_{highest_const_index} does.")
        for j, consts in enumerate(consts_values):
            if len(set(const_indices)) != len(consts):
                print(f"Warning in equation {id}, constants {j}: The number of constants does not match the number of constants in the equations.")

        for j, init in enumerate(init_values):
            if len(init) != dim:
                print(f"Error in equation {id}, init {j}: The number of initial values does not match the dimension of the equation.")
    print("VALIDATION DONE")


def process_equations(equations):
    """Create sympy expressions for each of the equations (and their different parameter values).
    We directly add the list of expressions to each dictionary.
    """
    validate_equations(equations)
    for eq_dict in equations:
        substituted_fns = create_substituted_functions(eq_dict)
        eq_dict['substituted'] = substituted_fns
    print("PROCESSING DONE")


def create_substituted_functions(eq_dict):
    """For a given equation, create sympy expressions where the different parameter values have been substituted in."""
    eq_string = eq_dict['eq']
    consts_values = eq_dict['consts']
    individual_eqs = eq_string.split('|')
    const_symbols = sp.symbols([f'c_{i}' for i in range(len(consts_values[0]))])
    parsed_eqs = [sp.sympify(eq) for eq in individual_eqs]

    substituted_fns = []
    for consts in consts_values:
        const_subs = dict(zip(const_symbols, consts))
        substituted_fns.append([eq.subs(const_subs) for eq in parsed_eqs])
    return substituted_fns

def solve_equations(equations, config):
    """Solve all equations for a given config.

    We add the solutions to each of the equations dictionary as a list of list of solution dictionaries.
    The list of list represents (number of parameter settings x number of initial conditions).
    """
    for eq_dict in tqdm(equations):
        eq_dict['solutions'] = []
        var_symbols = sp.symbols([f'x_{i}' for i in range(eq_dict['dim'])])
        for i, fns in enumerate(eq_dict['substituted']):
            eq_dict['solutions'].append([])
            callable_fn = lambda t, x: np.array([f(*x) for f in [sp.lambdify(var_symbols, eq, 'numpy') for eq in fns]])
            for initial_conditions in eq_dict['init']:
                sol = solve_ivp(callable_fn, **config, y0=initial_conditions)
                sol_dict = {
                    "success": sol.success,
                    "message": sol.message,
                    "t": sol.t.tolist(),
                    "y": sol.y.tolist(),
                    "nfev": int(sol.nfev),
                    "njev": int(sol.njev),
                    "nlu": int(sol.nlu),
                    "status": int(sol.status),
                }
                if sol.status != 0:
                    print(f"Error in equation {eq_dict['id']}: {eq_dict['eq_description']}, constants {i}, initial conditions {initial_conditions}: {sol.message}")
                sol_dict['consts'] = eq_dict['consts'][i]
                sol_dict['init'] = initial_conditions
                eq_dict['solutions'][i].append(sol_dict)
    print("SOLVING DONE")

def make_dataset(equation) -> OdeDataset:
    '''
    Make a dataset from an equation.

    Args:
        equation: Equation to make a dataset from.

    Returns:
        dataset: Dataset.
    '''


    sol = equation['solutions'][0][0]
    t = np.array(sol['t'])
    X = np.array(sol['y']).T

    # true time derivatives
    Y_true = np.zeros_like(X) # true time derivatives (n_time, dim)
    symbols = sp.symbols([f'x_{i}' for i in range(equation['dim'])])
    for i, equation_expr in enumerate(equation['substituted'][0]):
        fn = sp.lambdify(symbols, equation_expr, 'numpy')
        Y_true[:, i] = fn(*X.T).T
    return OdeDataset(t=t, u=X, du=Y_true)

def add_noise(dataset: OdeDataset, snr: float) -> OdeDataset:
    """Add noise to a dataset.

    Args:
        dataset: Dataset to add noise to.
        snr: Signal-to-noise ratio in dB.

    Returns:
        dataset: Dataset with noise added.
    """
    sigma2 = 10**(-snr/10)
    noise = np.random.randn(*dataset.u.shape) * np.sqrt(sigma2)
    trajectory = dataset.u * (1 + noise)
    new_dataset = OdeDataset(t=dataset.t, u=trajectory, du=dataset.du)
    return new_dataset

def save_dataset(dataset: OdeDataset, path: Path):
    """Save a dataset to disk.

    Args:
        dataset: Dataset to save.
        path: Path to the file to save the dataset to.
    """
    dataset = dataset._asdict()
    assert dataset['u'].shape == dataset['du'].shape
    assert dataset['t'].shape[0] == dataset['u'].shape[0]
    np.savez(path, **dataset)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',
                        type=Path,
                        default=Path('data/ode'),
                        help='Path to the directory to save the processed ODE equations and solutions'
                        )
    parser.add_argument('--snr',
                        type=list,
                        default=[40, 30, 20, 10],
                        help='Signal-to-noise ratio to add to the trajectories. Default is 0.0.'
                        )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    save_dir = args.save_dir

    equations = strogatz_ode.equations
    process_equations(equations)
    solve_equations(equations, config)
    for equation in equations:
        dataset_name = equation['name'].lower().replace(' ', '-')
        dataset = make_dataset(equation)
        save_path = save_dir / f'{dataset_name}' / f'{dataset_name}.npz'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dataset(dataset, save_path)
        for snr in args.snr:
            noisy_dataset = add_noise(dataset, snr)
            save_path = save_dir / f'{dataset_name}' /f'{dataset_name}_snr_{snr}.npz'
            save_dataset(noisy_dataset, save_path)
