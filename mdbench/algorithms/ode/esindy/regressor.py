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
    'poly_order': [1, 2, 3, 4],
    'optimizer_alpha': [1e-5, 1e-4],
    'n_models': [10, 20, 50],
    'n_subset_ratio': [0.5, 0.7, 0.9],
    'inclusion_probability_threshold': [0.2, 0.3, 0.4, 0.5]
}

class Estimator:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = None
        self.s = None # spatial grid
        self.function_names = None
        self.library_functions = None
        self.lib = None

        self.target_names = kwargs.get('target_names')
        self.basis_functions = kwargs.get('basis_functions', ['polynomial'])
        self.optimizer_threshold = kwargs.get('optimizer_threshold', 0.001)
        self.poly_order = kwargs.get('poly_order', 1)
        self.optimizer_alpha = kwargs.get('optimizer_alpha', 1e-5)
        self.optimizer_max_iter = 200
        self.n_models = kwargs.get('n_models', 20)
        self.n_subset_ratio = kwargs.get('n_subset_ratio', 0.5)
        self.inclusion_probability_threshold = kwargs.get('inclusion_probability_threshold', 0.2)

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
        self.lib = ps.feature_library.CustomLibrary(
            library_functions=self.library_functions,
            function_names=self.function_names,
            include_bias=True,
        )

        optimizer = ps.STLSQ(threshold=self.optimizer_threshold, alpha=self.optimizer_alpha, max_iter=self.optimizer_max_iter)
        model = ps.SINDy(feature_library=self.lib, optimizer=optimizer)
        n_samples = u_train.shape[0]
        n_subset = int(n_samples * self.n_subset_ratio)
        model.fit(
            u_train,
            x_dot=u_dot_train,
            t=t_train,
            ensemble=True, # with replacement is default, otherwise set replace=False
            n_models=self.n_models,
            n_subset=n_subset
        )
        self.model = model

    def predict(self, t_test, u_test):
        assert self.model is not None, "Model is not trained yet"
        inclusion_probs = np.count_nonzero(self.model.coef_list, axis=0) / self.n_models
        median_coefs = np.median(self.model.coef_list, axis=0)
        median_coefs[inclusion_probs < self.inclusion_probability_threshold] = 0
        features = self.lib.fit_transform(u_test)
        u_dot_predicted = features @ median_coefs.T
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

        system_dim = self.model.coef_list[0].shape[0]

        if self.target_names:
            target_names = self.target_names
        else:
            target_names = [f'u{i}' for i in range(system_dim)]

        def replace_variable(rhs, from_variable, to_variable):
            def replace_numbers(match):
                transformed = match.group(1)[1:].replace('1', 'x').replace('2', 'y').replace('3', 'z')
                return f"_{transformed}"
            rhs = rhs.replace(str(from_variable), str(to_variable))
            rhs = re.sub(r"(_[123]+)", replace_numbers, rhs)
            return rhs

        equations = []

        inclusion_probs = np.count_nonzero(self.model.coef_list, axis=0) / self.n_models
        coeffs = np.median(self.model.coef_list, axis=0)
        coeffs[inclusion_probs < self.inclusion_probability_threshold] = 0
        features = self.model.get_feature_names()

        for d in range(system_dim):
            rhs = ''
            dim_coeffs = coeffs[d]

            indices = np.abs(dim_coeffs) > 1e-10

            active_coeffs = dim_coeffs[indices]
            active_features = np.array(features)[indices]

            for i, (feature, coeff) in enumerate(zip(active_features, active_coeffs)):
                if i != 0:
                    rhs += ' + '
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
