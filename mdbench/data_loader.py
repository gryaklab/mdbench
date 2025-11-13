import sys
from pathlib import Path
from typing import List, Tuple, Union
from itertools import product

from scipy.io import loadmat
import numpy as np

if sys.version_info >= (3, 7):
    from findiff import FinDiff as Diff
else:
    from findiff import Diff

import json

def apply_diff(diff, u, order):
    result = u
    for _ in range(order):
        result = diff(result)
    return result


TEST_RATIO = 0.2
class PdeDataset:
    def __init__(self, name, t, s, u, du_true):
        '''
        s: a list of spatial coordinates. len(s) = spatial dimension
        t: time stamps (n_time)
        u: solution (n_space, n_time, dim)
        du: true time derivative of u (n_space, n_time, dim)
        '''
        if u.ndim == 2:
            u = np.expand_dims(u, 2)
        self.name = name
        self.t = t.ravel() # (n_time)
        self.s = s
        self.u = u # (n_time, *d_space, dim)
        self.du_true = du_true
        self.du = None
        self._compute_derivatives()


    def _compute_derivatives(self):
        dt = (self.t[1] - self.t[0]).item()
        t_index = self.u.ndim - 2
        diff = Diff(t_index, dt)
        self.du = diff(self.u)

    def get_s(self):
        return self.s

    def get_train(self):
        n_train = int(len(self.t)*(1-TEST_RATIO))
        t_train = self.t[:n_train]
        u_train = self.u[..., :n_train, :]
        du_train = self.du[..., :n_train, :]
        du_true_train = self.du_true[..., :n_train, :]
        return t_train, u_train, du_train, du_true_train

    def get_test(self):
        n_test = int(TEST_RATIO * len(self.t))
        t_test = self.t[-n_test:]
        u_test = self.u[..., -n_test:, :]
        du_test = self.du[..., -n_test:, :]
        du_true_test = self.du_true[..., -n_test:, :]
        return t_test, u_test, du_test, du_true_test

class SymbolicPdeDataset:
    '''PDE dataset with pre computed spatial derivatives as symbols'''
    def __init__(self, dataset: PdeDataset, derivative_order=4):
        self.name = dataset.name
        self.derivative_order = derivative_order

        self.t = dataset.t
        self.s = dataset.s
        self.u = dataset.u
        self.du_true = dataset.du_true
        self.dim = dataset.u.shape[-1]
        self.target_names = ['u{}'.format(i) for i in range(self.dim)]

        self.spatial_dim = len(self.s)

        self.symbols = None
        self.symbol_names = None # ['u0', 'u1', 'u0_x', 'u1_y', 'u0_{xx}', 'u1_{yy}', ...']
        self.targets = None
        self.targets_names = None # ['u', 'v', ...]

        self._compute_derivatives()

    def _compute_derivatives(self):
        dt = (self.t[1] - self.t[0]).item()
        dx = self.s[0][1] - self.s[0][0]
        dy = self.s[1][1] - self.s[1][0] if self.spatial_dim > 1 else None # 2D and 3D cases
        dz = self.s[2][1] - self.s[2][0] if self.spatial_dim > 2 else None # 3D case

        diff_t = Diff(self.spatial_dim, dt)
        diff_x = Diff(0, dx)
        diff_y = Diff(1, dy) if self.spatial_dim > 1 else None
        diff_z = Diff(2, dz) if self.spatial_dim > 2 else None

        symbols = [self.u.reshape(-1, self.dim)]
        symbol_names = [f'{target_name}' for target_name in self.target_names]

        for d_order in range(1, self.derivative_order+1):
            u_xd = apply_diff(diff_x, self.u, d_order)
            symbols.append(u_xd.reshape(-1, self.dim))
            for i in range(self.dim):
                symbol_names.append(f'{self.target_names[i]}_{"x"*d_order}')

            if self.spatial_dim > 1: # 2D and 3D cases
                u_yd = apply_diff(diff_y, self.u, d_order)
                symbols.append(u_yd.reshape(-1, self.dim))
                for i in range(self.dim):
                    symbol_names.append(f'{self.target_names[i]}_{"y"*d_order}')

        if self.spatial_dim > 2: # 3D case
            u_zd = (diff_z)(self.u) # only one first order derivative is computed for 3D case
            symbols.append(u_zd.reshape(-1, self.dim))
            for i in range(self.dim):
                symbol_names.append(f'{self.target_names[i]}_{"z"*d_order}')

        self.symbol_names = symbol_names
        self.symbols = np.concatenate(symbols, axis=-1)
        self.targets_names = [f'{target_name}_t' for target_name in self.target_names]
        self.targets = diff_t(self.u).reshape(-1, self.dim)

    def get_train(self):
        '''
        Returns:
            t: (n_train)
            u: (n_train, (derivative_order+1)*dim)
            du: (n_train, dim)
            du_true: (n_train, dim)
        '''
        n_train = int(self.symbols.shape[0]*(1-TEST_RATIO))
        t_train = self.t[:n_train]
        u_train = self.symbols[:n_train]
        du_train = self.targets[:n_train]
        du_true_train = self.du_true.reshape(-1, self.dim)[:n_train]
        return t_train, u_train, du_train, du_true_train

    def get_test(self):
        '''
        Returns:
            t: (n_test)
            u: (n_test, (derivative_order+1)*dim)
            du: (n_test, dim)
            du_true: (n_test, dim)
        '''
        n_test = int(self.symbols.shape[0]*TEST_RATIO)
        t_test = self.t[-n_test:]
        u_test = self.symbols[-n_test:]
        du_test = self.targets[-n_test:]
        du_true_test = self.du_true.reshape(-1, self.dim)[-n_test:]
        return t_test, u_test, du_test, du_true_test

class OdeDataset:
    def __init__(self, name, t, u, du_true):
        '''
        t: time stamps (n_time)
        u: observed trajectories (n_time, dim)
        du_true: true time derivatives (n_time, dim)
        '''
        self.name = name
        self.t = t
        self.u = u
        self.du_true = du_true
        self.du = None
        self.target_names = ['u{}'.format(i) for i in range(self.u.shape[-1])]
        self.symbol_names = self.target_names
        self._compute_derivatives()

    def _compute_derivatives(self):
        dt = (self.t[1] - self.t[0]).item()
        diff = Diff(0, dt)
        self.du = diff(self.u)

    def get_train(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Returns:
            t: time stamps (n_train)
            X: observed state variables (n_train, dim)
            Y_true: true time derivatives (n_train, dim)
            Y_approx: time derivatives approximated with finite difference (n_train, dim)
        '''
        n_train = int(len(self.t)*(1-TEST_RATIO))
        t = self.t[:n_train]
        u_train = self.u[:n_train]
        du_train = self.du[:n_train]
        du_true_train = self.du_true[:n_train]
        return t, u_train, du_train, du_true_train

    def get_test(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Returns:
            t: time stamps (n_test)
            u: observed state variables (n_test, dim)
            du_true: true time derivatives (n_test, dim)
            du: time derivatives approximated with finite difference (n_test, dim)
        '''
        n_test = int(len(self.t)*TEST_RATIO)
        t = self.t[-n_test:]
        u_test = self.u[-n_test:]
        du_test = self.du[-n_test:]
        du_true_test = self.du_true[-n_test:]
        return t, u_test, du_test, du_true_test

def load_pde_dataset(path: List[Path], convert_to_symbolic: bool = False):
    data = np.load(path)
    name = path.stem
    t = data['t']
    u = data['u']
    du_true = data['du']

    s = [data['x']]
    if 'y' in data:
        s.append(data['y'])
    if 'z' in data:
        s.append(data['z'])

    dataset = PdeDataset(name=name, t=t, s=s, u=u, du_true=du_true)
    if convert_to_symbolic:
        dataset = SymbolicPdeDataset(dataset)
    return dataset

def load_ode_dataset(path: Path) -> OdeDataset:
    '''
    Load the dataset from the npz file.

    Args:
        path: Path to the JSON file.

    Returns:
        datasets: OdeDataset.
    '''
    data = np.load(path)
    name = path.stem
    t = data['t']
    u = data['u']
    du_true = data['du']
    return OdeDataset(name=name, t=t, u=u, du_true=du_true)

def load_datasets(data_paths: List[Path], algorithm_type: str, data_type: str):
    convert_to_symbolic = algorithm_type == 'sr'
    pde_loader = lambda path: load_pde_dataset(path, convert_to_symbolic)
    loader_func = pde_loader if data_type == 'pde' else load_ode_dataset
    for data_path in data_paths:
        yield loader_func(data_path)
