import os
from itertools import product
from pathlib import Path
import tempfile

import numpy as np
import torch
import sympy as sp
from findiff import Diff


from mdbench.utils import str_to_sympy

hyper_params = {
    'poly_order': [1, 2, 3, 4],
    'diff_order': [1, 2, 3, 4],
    'hidden_layers': [
        [50, 50, 50, 50]
    ],
    'learning_rate': [1e-3],
    'threshold': [0.1, 0.3, 0.5],
}

# DeepMoD functions

from deepymod import DeepMoD
from deepymod.data import Dataset, get_train_test_loader
from deepymod.data.samples import Subsample_random
from deepymod.model.func_approx import NN
from deepymod.model.library import Library1D, Library2D
from deepymod.model.constraint import LeastSquares
from deepymod.model.sparse_estimators import Threshold
from deepymod.training import train
from deepymod.training.sparsity_scheduler import TrainTestPeriodic

# Settings for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuring GPU or CPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class Estimator:
    def __init__(self, **kwargs):
        self.model = None
        self.s = None # spacial grid
        self.spatial_dim = None
        self.system_dim = None
        self.poly_order = kwargs.get('poly_order', 2)
        self.diff_order = kwargs.get('diff_order', 3)
        self.hidden_layers = kwargs.get('hidden_layers', [50, 50, 50, 50])
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.threshold = kwargs.get('threshold', 0.1)
        self.target_names = kwargs.get('target_names', None)
        self.log_dir = self._get_log_dir()

    def _get_log_dir(self):
        '''A temporary dir for the log file which is unique for each set of hyperparameters'''
        return str(Path(tempfile.gettempdir()) / f'deepmod_{self.poly_order}_{self.diff_order}_{self.hidden_layers}')

    def set_spatial_grid(self, s):
        '''Set the spatial grid '''
        self.s = s
        self.spatial_dim = len(s)

    def fit(self, t_train, u_train, u_dot_train):
        # torch.cuda.empty_cache()
        self.system_dim = u_train.shape[-1]
        if self.spatial_dim == 1:
            # 1D case: (x, t, u) -> (t, x, u)
            u_train = np.transpose(u_train, (1, 0, 2))
        elif self.spatial_dim == 2:
            if self.poly_order > 1:
                raise ValueError("2D case with poly_order > 1 is not supported")
            # 2D case: (x, y, t, u) -> (t, x, y, u)
            u_train = np.transpose(u_train, (2, 0, 1, 3))
        else:
            raise ValueError(f"Number of spatial dimensions must be 1 or 2, got {self.spatial_dim}")

        def create_data():
            if self.spatial_dim == 1:
                x, t = np.meshgrid(self.s[0], t_train)
                coords = np.stack((t, x), axis=-1)
            elif self.spatial_dim == 2:
                x, y, t = np.meshgrid(self.s[0], self.s[1], t_train)
                coords = np.stack((t, x, y), axis=-1)
            coords = torch.from_numpy(coords).float()
            u = torch.from_numpy(u_train).float()
            return coords, u

        dataset = Dataset(
            create_data,
            preprocess_kwargs={
                "noise_level": 0.0,
                "normalize_coords": False,
                "normalize_data": False,
            },
            subsampler=Subsample_random,
            subsampler_kwargs={"number_of_samples": 2000},
            device=device,
        )

        train_dataloader, test_dataloader = get_train_test_loader(dataset, train_test_split=0.8)

        if self.spatial_dim == 1:
            n_input = 2 # time, x
            library = Library1D(poly_order=self.poly_order, diff_order=self.diff_order)
        elif self.spatial_dim == 2:
            n_input = 3 # time, x, y
            library = Library2D(poly_order=self.poly_order)
        n_output = u_train.shape[-1] # u
        n_hidden = self.hidden_layers
        network = NN(n_input, n_hidden, n_output)

        estimator = Threshold(self.threshold)
        sparsity_scheduler = TrainTestPeriodic(periodicity=50, patience=10, delta=1e-5)
        constraint = LeastSquares()
        model = DeepMoD(network, library, estimator, constraint).to(device)
        # Defining optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), betas=(0.99, 0.99), amsgrad=True, lr=self.learning_rate
        )
        train(
            model,
            train_dataloader,
            test_dataloader,
            optimizer,
            sparsity_scheduler,
            log_dir=self.log_dir,
            max_iterations=50_000,
        )
        self.model = model

    def predict(self, t_test, u_test):
        assert self.model is not None, "Model is not trained yet"
        torch.cuda.empty_cache()
        symbol_values = []
        symbol_names = []

        dx = self.s[0][1] - self.s[0][0]
        diff_x = Diff(0, dx)
        dy = self.s[1][1] - self.s[1][0] if self.spatial_dim == 2 else None
        diff_y = Diff(1, dy) if dy is not None else None
        if self.spatial_dim == 1:
            u_xs = [(diff_x**(i+1))(u_test) for i in range(self.diff_order)]
            for i in range(self.system_dim):
                symbol_names.append(f'u{i}')
                symbol_values.append(u_test[..., i])
                for j in range(self.diff_order):
                    symbol_names.append(f"u{i}_{'x'*(j+1)}")
                    symbol_values.append(u_xs[j][..., i])
        elif self.spatial_dim == 2:
            u_x = diff_x(u_test)
            u_y = diff_y(u_test)
            u_xx = (diff_x**2)(u_test)
            u_yy = (diff_y**2)(u_test)
            u_xy = (diff_x*diff_y)(u_test)

            for i in range(self.system_dim):
                symbol_names.append(f'u{i}')
                symbol_values.append(u_test[..., i])
                symbol_names.append(f"u{i}_x")
                symbol_values.append(u_x[..., i])
                symbol_names.append(f"u{i}_y")
                symbol_values.append(u_y[..., i])
                symbol_names.append(f"u{i}_xx")
                symbol_values.append(u_xx[..., i])
                symbol_names.append(f"u{i}_yy")
                symbol_values.append(u_yy[..., i])
                symbol_names.append(f"u{i}_xy")
                symbol_values.append(u_xy[..., i])
        symbol_values = np.stack(symbol_values, axis=-1)
        symbol_values = symbol_values.reshape(-1, len(symbol_names))
        symbols = [sp.Symbol(ss) for ss in symbol_names]

        equations = self.to_sympy()
        u_dot_predicted = []
        for equation in equations:
            eq_func = sp.lambdify(symbols, equation.rhs, modules='numpy')
            u_dot_predicted.append(eq_func(*symbol_values.T))
        return np.reshape(u_dot_predicted, u_test.shape)

    def complexity(self):
        '''Return the complexity of the model defined as the number of operations in the equation'''
        assert self.model is not None, "Model is not trained yet"
        equations = self.to_sympy()
        complexity = 0
        for equation in equations:
            for arg in sp.preorder_traversal(equation.rhs):
                complexity += 1
        return complexity

    def to_str(self):
        target_names = [f'u{i}' for i in range(self.system_dim)]
        equations = []
        all_coefs = self.model.constraint_coeffs()
        if len(target_names) == 1 and self.spatial_dim == 1:
            target = target_names[0]
            diffs = ['1'] + [f'{target}_{"x"*d}' for d in range(1, self.diff_order+1)]
            polys = ['1'] + [f'{target}^{p}' for p in range(1, self.poly_order + 1)]
            all_library_terms = [f"{m}*{d}" for m, d in product(polys, diffs)]
        elif len(target_names) == 2 and self.spatial_dim == 1:
            diffs = [
                [''] + [f'{target}_{"x"*d}' for d in range(1, self.diff_order+1)] for target in target_names
            ]
            polys = [
                [''] + [f'{target}^{p}' for p in range(1, self.poly_order + 1)] for target in target_names
            ]
            uv = ['*'.join((x, y)) for x, y in product(*polys)]
            dudv = ['*'.join((x, y)) for x, y in product(*diffs)][1:]
            u_dv = ['*'.join((x, y)) for p, d in product(polys, diffs) for x, y in product(p[1:], d[1:])]
            all_library_terms = uv + dudv + u_dv
        elif len(target_names) == 1 and self.spatial_dim == 2:
            target = target_names[0]
            diffs = ['1', f'{target}_x', f'{target}_y', f'{target}_xx', f'{target}_yy', f'{target}_xy']
            polys = ['1'] + [f'{target}^{p}' for p in range(1, self.poly_order + 1)]
            all_library_terms = [f"{m}*{d}" for m, d in product(polys, diffs)]
        else:
            raise ValueError(
                f"Number of target names must be 1 or 2 and number of"
                f"spatial dimensions must be 1 or 2, got {len(target_names)} and {self.spatial_dim  }")
        for target_index, target in enumerate(target_names):
            rhs = ''
            coefs = all_coefs[target_index][:, 0].detach().cpu().numpy()
            mask = abs(coefs) > 1e-10
            library_terms = np.array(all_library_terms)[mask]
            coefs = coefs[mask]
            for i, (coef, term) in enumerate(zip(coefs, library_terms)):
                if i != 0:
                    rhs += ' + '
                rhs += f"{coef}*{term}"
            equation = f'{target} = {rhs}'
            equations.append(equation)
        return '\n'.join(equations)

    def to_sympy(self):
        assert self.model is not None, "Model is not trained yet"
        equations = self.to_str()
        return str_to_sympy(equations)
