
import re
from typing import List

import numpy as np
import pysindy as ps
import sympy as sp

from mdbench.utils import str_to_sympy

hyper_params = {
    'optimizer_threshold': np.logspace(-7, 0, 16),
    'basis_functions': [
        ['polynomial'],
        ['polynomial', 'sin', 'cos'],
    ],
    'poly_order': [1, 2, 3, 4],
    'optimizer_alpha': [1e-5, 1e-4],
}

class Estimator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None
        self.s = None # spatial grid
        self.function_names = None
        self.library_functions = None
        self.target_names = kwargs.get('target_names', None)
        self.basis_functions = kwargs.get('basis_functions', ['polynomial'])
        self.optimizer_threshold = kwargs.get('optimizer_threshold', 0.001)
        self.optimizer_alpha = kwargs.get('optimizer_alpha', 1e-5)
        self.poly_order = kwargs.get('poly_order', 1)
        self.optimizer_max_iter = 200

    def _set_library_functions(self):
        self.library_functions = []
        self.function_names = []
        for basis_function in self.basis_functions:
            if basis_function == 'polynomial':
                self.library_functions.extend([(lambda p: lambda x: x**p)(i) for i in range(1, self.poly_order + 1)])
                self.function_names.extend([(lambda p: lambda x: f'{x}^{p}')(i) for i in range(1, self.poly_order + 1)])
            elif basis_function == 'sin':
                self.library_functions.extend([lambda x: np.sin(x)])
                self.function_names.extend([lambda x: f'sin({x})'])
            elif basis_function == 'cos':
                self.library_functions.extend([lambda x: np.cos(x)])
                self.function_names.extend([lambda x: f'cos({x})'])
            elif basis_function == 'exp':
                self.library_functions.extend([lambda x: np.exp(x)])
                self.function_names.extend([lambda x: f'exp({x})'])


    def fit(self, t_train, X_train, Y_train):
        self._set_library_functions()
        lib = ps.feature_library.CustomLibrary(
            library_functions=self.library_functions,
            function_names=self.function_names,
            include_bias=True,
        )
        optimizer = ps.STLSQ(threshold=self.optimizer_threshold, alpha=self.optimizer_alpha, max_iter=self.optimizer_max_iter)
        model = ps.SINDy(feature_library=lib, optimizer=optimizer)
        model.fit(X_train, x_dot=Y_train, t=t_train)
        self.model = model

    def predict(self, t_test, X_test):
        assert self.model is not None, "Model is not trained yet"
        Y_predicted = self.model.predict(X_test)
        return Y_predicted

    def complexity(self):
        '''Return the complexity of the model defined as the number of operations in the equation'''
        assert self.model is not None, "Model is not trained yet"
        equations = self.to_sympy()
        complexity = 0
        for equation in equations:
            for arg in sp.preorder_traversal(equation.rhs):
                complexity += 1
        return complexity

    def to_str(self) -> str:
        '''Construct the equation string from the coefficient matrix'''
        assert self.model is not None, "Model is not trained yet"
        equations = []
        system_dim = len(self.model.equations())
        if self.target_names is None:
            self.target_names = [f'u{i}' for i in range(system_dim)]
        for d in range(system_dim):
            rhs = ''
            coeffs = self.model.coefficients()
            coeffs = coeffs[d]
            features = self.model.get_feature_names()
            indices = np.abs(coeffs) > 1e-10
            coeffs = coeffs[indices]
            features = np.array(features)[indices]
            for i, (feature, coeff) in enumerate(zip(features, coeffs)):
                if i != 0:
                    rhs += ' + '
                rhs += f'{coeff}*{feature}'

            lhs = f'{self.target_names[d]}_t'
            for i in range(system_dim):
                rhs = rhs.replace(f'x{i}', self.target_names[i])
            equations.append(f'{lhs} = {rhs}')

        return '\n'.join(equations)

    def to_sympy(self) -> List[sp.Equality]:
        assert self.model is not None, "Model is not trained yet"
        equations = self.to_str()
        return str_to_sympy(equations)