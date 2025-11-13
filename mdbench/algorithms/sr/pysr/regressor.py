
from multiprocessing import cpu_count

import numpy as np
np.random.seed(42)
import pysr

import mdbench.data_loader as data_loader
import mdbench.metrics as metrics

class Estimator:
    def __init__(self, **kwargs):
        self.model = None
        self.symbol_names = kwargs.get('symbol_names', None)
        self.target_names = kwargs.get('target_names', None)

        n_jobs = kwargs.get('n_jobs', -1)
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        self.model = pysr.PySRRegressor(
            procs=n_jobs,
            niterations=100,
            ncycles_per_iteration=1_000,
            population_size=100,
            populations=n_jobs,
            # timeout_in_seconds=2*60*60 - 10*60, # No timeout here
            maxsize=40,
            maxdepth=20,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["sin", "exp", "log", "sqrt"],
            constraints={
                **dict(
                    sin=9,
                    exp=9,
                    log=9,
                    sqrt=9,
                ),
                **{"/": (-1, 9)}
            },
            nested_constraints=dict(
                sin=dict(
                    sin=0,
                    exp=1,
                    log=1,
                    sqrt=1,
                ),
                exp=dict(
                    exp=0,
                    log=0,
                ),
                log=dict(
                    exp=0,
                    log=0,
                ),
                sqrt=dict(
                    sqrt=0,
                )
            ),
            parallelism='multiprocessing',
            batching=True,
            batch_size=128,
            weight_optimize=0.001,
            adaptive_parsimony_scaling=1_000.0,
            temp_equation_file=True,
        )

    def fit(self, t_train, X_train, Y_train):
        self.model.fit(X_train, Y_train, variable_names=self.symbol_names)
        system_equations = self.model.equations_
        if not isinstance(system_equations, list):
            system_equations = [system_equations]
        self.best_equations = []
        for i in range(len(system_equations)):
            equations = system_equations[i]
            equations['nmse'] = [metrics.nmse(Y_train[:, i], eq(X_train)) for eq in equations.lambda_format]
            equations['fitness'] = metrics.fitness(equations['nmse'], equations['complexity'])
            self.best_equations.append(equations.sort_values("fitness", ascending=False).iloc[0])


    def predict(self, t_test, X_test):
        assert self.model is not None, "Model is not trained yet"
        Y_predicted = []
        for eq in self.best_equations:
            Y_predicted.append(eq.lambda_format(X_test))
        return np.array(Y_predicted).T

    def complexity(self):
        '''Return the complexity of the model defined as the number of operations in the equation'''
        assert self.model is not None, "Model is not trained yet"
        return sum([eq['complexity'] for eq in self.best_equations])

    def to_str(self) -> str:
        assert self.model is not None, "Model is not trained yet"
        equations = []
        system_dim = len(self.best_equations)
        target_names = ['u{}'.format(i) for i in range(system_dim)]
        for lhs, eq in zip(target_names, self.best_equations):
            rhs = str(eq.sympy_format)
            for i in range(len(self.symbol_names)):
                rhs = rhs.replace(f'x_{{{i}}}', self.symbol_names[i])
            equations.append(f'{lhs}_t = {rhs}')
        return '\n'.join(equations)
