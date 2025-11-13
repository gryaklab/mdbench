import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as np
import sympy as sp
from eql.est import EQL

hyper_params = {
    "n_iter": (10_000,),
    "reg": (1e-4, 1e-3, 1e-2, 5e-2),
    "n_layers": (1, 2),
    "functions": (
        "id;mul;cos;sin;exp;square;sqrt;id;mul;cos;sin;exp;square;sqrt;log",
        "id;mul;cos;div;sqrt;cos;sin;div;mul;mul;cos;id;log",
    ),
}

class Estimator:
    def __init__(self, symbol_names=None, target_names=None, **kwargs):
        self.symbol_names = symbol_names
        self.target_names = target_names
        self.kwargs = kwargs
        self.model = EQL(random_state=42, **self.kwargs)

    def fit(self, t_train, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, t_test, X_test):
        y_pred = self.model.predict(X_test)
        y_pred = np.array(y_pred)
        return y_pred

    def to_str(self):
        equations = self.model.get_eqn()
        for i in range(len(equations)):
            target_name = self.target_names[i]
            equation = f'{target_name}_t = {equations[i]}'
            equation = equation.replace(f'x_{{{i}}}', self.symbol_names[i])
            equations[i] = equation
        return '\n'.join(str(equation) for equation in equations)

    def complexity(self):
        equations = self.model.get_eqn()
        complexity = 0
        for equation in equations:
            for _ in sp.preorder_traversal(equation):
                complexity += 1
        return complexity