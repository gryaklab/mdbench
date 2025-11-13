import sympy as sp
import numpy as np
from multiprocessing import cpu_count

from pyoperon.sklearn import SymbolicRegressor

import mdbench.metrics as metrics

default_hyper_params = {
    'allowed_symbols': "add,sub,mul,aq,sin,constant,variable",
    'brood_size': 10,
    'comparison_factor': 0,
    'crossover_internal_probability': 0.9,
    'crossover_probability': 1.0,
    'epsilon': 1e-05,
    'female_selector': "tournament",
    'generations': 1000,
    'initialization_max_depth': 5,
    'initialization_max_length': 10,
    'initialization_method': "btc",
    'irregularity_bias': 0.0,
    'local_search_probability': 1.0,
    'lamarckian_probability': 1.0,
    'optimizer_iterations': 1,
    'optimizer': 'lm',
    'male_selector': "tournament",
    'max_depth': 10,
    'max_evaluations': 1000000,
    'max_length': 50,
    'max_selection_pressure': 100,
    'model_selection_criterion': "minimum_description_length",
    'mutation_probability': 0.25,
    'objectives': [ 'r2', 'length' ],
    'offspring_generator': "os",
    'pool_size': 1000,
    'population_size': 1000,
    'random_state': 42,
    'reinserter': "keep-best",
    'max_time': 43200,
    'tournament_size':3,
    'add_model_intercept_term': True,
    'add_model_scale_term': True
}

class Estimator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.s = None # spatial grid
        self.dim = None # number of equations in the system
        self.models = None # list of models
        self.symbol_names = kwargs.get('symbol_names', None)
        self.target_names = kwargs.get('target_names', None)
        self._complexity = None
        self.n_jobs = kwargs.get('n_jobs', -1)
        self.n_jobs = cpu_count() if self.n_jobs == -1 else self.n_jobs

    def set_spatial_grid(self, s):
        self.s = s

    def fit(self, t_train, u_train, u_dot_train):
        self._complexity = 0
        self.models = []
        self.dim = u_dot_train.shape[-1]
        u_train = np.array(u_train, dtype=np.float64, order='F')
        for i in range(self.dim):
            model = SymbolicRegressor(**default_hyper_params, n_threads=self.n_jobs)
            u_dot_train_column = u_dot_train[:, i]
            model.fit(u_train, u_dot_train_column)

            pareto_front = model.pareto_front_
            u_dot_preds = [model.evaluate_model(solution['tree'], u_train) for solution in pareto_front]
            nmses = [metrics.nmse(u_dot_train_column, u_dot_pred) for u_dot_pred in u_dot_preds]
            complexities = [solution['complexity'] for solution in pareto_front]
            scores = [metrics.fitness(nmse, complexity) for nmse, complexity in zip(nmses, complexities)]
            best_idx = np.argmax(scores)

            model.model_ = pareto_front[best_idx]['tree']
            self._complexity += pareto_front[best_idx]['complexity']
            self.models.append(model)

    def predict(self, t_test, u_test):
        u_test = np.array(u_test, dtype=np.float64, order='F')
        u_dot_pred = []
        for i in range(self.dim):
            u_dot_pred.append(self.models[i].predict(u_test))
        return np.array(u_dot_pred).T

    def to_str(self):
        equations = []
        for i in range(self.dim):
            equation_str = self.models[i].get_model_string(self.models[i].model_, precision=10, names=self.symbol_names)
            equation_str = equation_str.replace('^', '**')
            equation_str = sp.parse_expr(equation_str)
            equations.append(f'{self.target_names[i]}_t = {equation_str}')
        return '\n'.join(equations)

    def complexity(self):
        return self._complexity
