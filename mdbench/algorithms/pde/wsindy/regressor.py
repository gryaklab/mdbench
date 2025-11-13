
import re

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
    'n_integration_points': [200, 2000],
    'derivative_order': [1, 2, 3, 4],
    'poly_order': [1, 2, 3, 4],
}

class Estimator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None
        self.s = None # spatial grid
        self.function_names = None
        self.library_functions = None

        self.basis_functions = kwargs.get('basis_functions', ['polynomial'])
        self.optimizer_threshold = kwargs.get('optimizer_threshold', 0.001)
        self.derivative_order = kwargs.get('derivative_order', 1)
        self.poly_order = kwargs.get('poly_order', 1)
        self.n_integration_points = kwargs.get('n_integration_points', 10)

        self.optimizer_max_iter = 200

    def set_spatial_grid(self, s):
        '''Set the spatial grid '''
        self.s = s

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

    def fit(self, t_train, u_train, u_dot_train):
        self._set_library_functions()
        ST = np.stack(np.meshgrid(*(*self.s, t_train), indexing='ij'), axis=-1)
        pde_lib = ps.WeakPDELibrary(
            library_functions=self.library_functions,
            function_names=self.function_names,
            derivative_order=self.derivative_order,
            include_bias=True,
            spatiotemporal_grid=ST,
            is_uniform=True,
            K=self.n_integration_points,
        )

        optimizer = ps.SR3(threshold=self.optimizer_threshold, max_iter=self.optimizer_max_iter)
        model = ps.SINDy(feature_library=pde_lib, optimizer=optimizer)
        model.fit(u_train)
        self.model = model

    def predict(self, t_test, u_test):
        assert self.model is not None, "Model is not trained yet"
        function_names = self.model.feature_library.function_names
        functions = self.model.feature_library.functions
        derivative_order = self.model.feature_library.derivative_order
        s = np.stack(np.meshgrid(*self.s, indexing='ij'), axis=-1)
        exact_lib = ps.PDELibrary(
            library_functions=functions,
            function_names=function_names,
            derivative_order=derivative_order,
            spatial_grid=s,
            include_bias=True,
        )
        features = exact_lib.fit_transform(u_test)
        coeffs = self.model.coefficients()
        u_dot_predicted = features @ coeffs.T
        return u_dot_predicted

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
        assert self.model is not None, "Model is not trained yet"
        system_dim = len(self.model.equations())
        target_names = [f'u{i}' for i in range(system_dim)]

        def replace_variable(rhs, from_variable, to_variable):
            def replace_numbers(match):
                transformed = match.group(1)[1:].replace('1', 'x').replace('2', 'y').replace('3', 'z')
                return f"_{transformed}"
            rhs = rhs.replace(str(from_variable), str(to_variable))
            rhs = re.sub(r"(_[123]+)", replace_numbers, rhs)
            return rhs

        equations = []
        for d in range(system_dim):
            rhs = ''
            coeffs = self.model.coefficients()
            features = self.model.get_feature_names()

            coeffs = coeffs[d]
            indices = np.abs(coeffs) > 1e-10

            coeffs = coeffs[indices]
            features = np.array(features)[indices]
            for i, (feature, coeff) in enumerate(zip(features, coeffs)):
                if i != 0:
                    rhs += ' + '
                if feature.count('x') == 2:
                    first_x = feature.find('x')
                    second_x = feature.find('x', first_x + 1)
                    feature = f'({feature[:second_x]})*({feature[second_x:]})'
                rhs += f'{coeff}*{feature}'

            lhs = f'{target_names[d]}_t'
            for i in range(system_dim):
                rhs = replace_variable(rhs, f'x{i}', target_names[i])
            equations.append(f'{lhs} = {rhs}')

        return '\n'.join(equations)

    def to_sympy(self):
        assert self.model is not None, "Model is not trained yet"
        equations = self.to_str()
        return str_to_sympy(equations)
