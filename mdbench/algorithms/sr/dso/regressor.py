from dso import DeepSymbolicOptimizer
import numpy as np
from numpy.random import default_rng
import os
import json
import pandas as pd
from mdbench import metrics
from dso.program import from_str_tokens
import re
import sympy as sp
from mdbench.utils import str_to_sympy

class Estimator:
    """DSO estimator with GP-MELD integrated into the mdbench evaluation framework"""
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.s = None # spatial grid
        self.dim = None # number of equations in the system
        self.models = None # list of models
        self.symbol_names = kwargs.get('symbol_names', None)
        self.target_names = kwargs.get('target_names', None)
        self.best_equations = []
        
        # load config file with the hyperparameters optimized by the DSO authors
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, 'dso_config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
    def set_spatial_grid(self, s):
        self.s = s
        
    def fit(self, t_train, u_train, u_dot_train):      
        # sampling data to avoid large matrix memory error within the dso algorithm
        sample_size = 10000
        if len(u_train) > sample_size:
            rng = default_rng()
            indices = rng.choice(len(u_train), sample_size, replace=False)
            u_train = u_train[indices]
            u_dot_train = u_dot_train[indices]
        
        self.dim = u_dot_train.shape[-1]
        u_train = np.array(u_train, dtype=np.float64)
        for i in range(self.dim):
            try:
                run_config = self.config.copy()
                model = DeepSymbolicOptimizer(config=run_config)
                u_dot_train_column = u_dot_train[:, i]
                model.config_task['dataset'] = (u_train, u_dot_train_column)
                
                model.train()
                
                # read the Pareto front data from the log file
                log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'log', 'dso_run')
                pareto_file = os.path.join(log_dir, 'dso_regression_42_pf.csv')
                
                if os.path.exists(pareto_file):
                    pareto_data = pd.DataFrame(pd.read_csv(pareto_file))
                    for idx, row in pareto_data.iterrows():
                        try:
                            program = from_str_tokens(row['traversal'])
                            pred = program.execute(u_train)
                            nmse = metrics.nmse(u_dot_train_column, pred)
                            complexity = len(row['traversal'].split(','))
                            pareto_data.loc[idx, 'nmse'] = nmse
                            pareto_data.loc[idx, 'fitness'] = metrics.fitness(nmse, complexity)
                        except Exception as e:
                            pareto_data.loc[idx, 'fitness'] = float('-inf')
                    
                    # store the equation with the highest fitness score
                    best_equation = pareto_data.sort_values('fitness', ascending=False).iloc[0]
                    self.best_equations.append(best_equation)
            except Exception as e:
                print(f"An error occurred during model fitting: {e}")
                raise

    def predict(self, t_test, u_test):
        assert self.best_equations is not None, "Models are not trained yet"
        u_test = np.array(u_test, dtype=np.float64, order='F')
        u_dot_pred = []
        for equation in self.best_equations:
            try:
                from dso.program import from_str_tokens
                program = from_str_tokens(equation['traversal'])
                y_pred = program.execute(u_test)
                u_dot_pred.append(y_pred)
            except Exception as e:
                raise
        return np.array(u_dot_pred).T
    
    def complexity(self):
        assert self.best_equations is not None, "Models are not trained yet"
        equations = self.to_sympy()
        total_complexity = 0
        for eq in equations:
            if hasattr(eq, 'rhs'):
                # Traverse the right-hand side of the equation
                for _ in sp.preorder_traversal(eq.rhs):
                    total_complexity += 1
            else:
                # Handle cases where it's just an expression, not an equation
                for _ in sp.preorder_traversal(eq):
                    total_complexity += 1
        return total_complexity
    
    def to_str(self):
        assert self.best_equations is not None, "Models are not trained yet"
        equations = []
        for i, equation in enumerate(self.best_equations):
            if self.target_names is not None:
                lhs = f'{self.target_names[i]}_t'
            else:
                lhs = f'u{i}_t'
            
            rhs = equation['expression']
            if self.symbol_names:
                def replace_variable(match):
                    """extract the index from the matched variable, e.g., '5' from 'x5'
                    convert from DSO's 1-based index to our 0-based list index"""
                    var_index = int(match.group(1))
                    symbol_index = var_index - 1
                    if 0 <= symbol_index < len(self.symbol_names):
                        return self.symbol_names[symbol_index]
                    # else return it unchanged
                    return match.group(0)
                rhs = re.sub(r'\bx(\d+)\b', replace_variable, rhs)
            equations.append(f'{lhs} = {rhs}')
        
        return '\n'.join(equations)
    
    def to_sympy(self):
        assert self.best_equations is not None, "Models are not trained yet"
        equations_str = self.to_str()
        return str_to_sympy(equations_str)