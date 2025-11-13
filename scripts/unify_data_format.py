from pathlib import Path
from dataclasses import dataclass
import argparse

import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from findiff import Diff

np.random.seed(42)

@dataclass
class PdeData:
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    du: np.ndarray
    y: np.ndarray = None
    z: np.ndarray = None
    '''
    t: time points
    x: spatial points on x-axis
    y: spatial points on y-axis. Optional (only for 2D and 3D systems)
    z: spatial points on z-axis. Optional (only for 3D systems)
    u: observed solution
    du: true derivative of u
    '''

@dataclass
class OdeData:
    t: np.ndarray
    u: np.ndarray
    du: np.ndarray
    '''
    t: time points
    u: solution
    du: true derivative of u
    '''

def load_burgers(data_path: Path) -> PdeData:
    data = loadmat(data_path)
    t = data['t'].ravel()
    dt = t[1] - t[0]
    x = data['x'].ravel()
    u = np.real(data['usol'])
    u = np.expand_dims(u, axis=-1)
    du = Diff(1, dt)(u)
    return PdeData(t=t, x=x, y=None, z=None, u=u, du=du)

def load_kdv(data_path: Path) -> PdeData:
    data = loadmat(data_path)
    t = data['t'].ravel()
    dt = t[1] - t[0]
    x = data['x'].ravel()
    u = np.real(data['usol'])
    u = np.expand_dims(u, axis=-1)
    du = Diff(u.ndim - 2, dt)(u)
    return PdeData(t=t, x=x, y=None, z=None, u=u, du=du)

def load_kuramoto_sivishinky(data_path: Path) -> PdeData:
    data = loadmat(data_path)
    t = data['tt'].ravel()
    dt = t[1] - t[0]
    x = data['x'].ravel()
    u = np.real(data['uu'])
    u = np.expand_dims(u, axis=-1)
    du = Diff(u.ndim - 2, dt)(u)
    return PdeData(t=t, x=x, y=None, z=None, u=u, du=du)

def load_advection1d(data_path: Path) -> PdeData:
    data = np.load(data_path)
    t = data['t'][:-1]
    dt = t[1] - t[0]
    x = data['x'].ravel()
    u = data['u'].T
    u = np.expand_dims(u, axis=-1)
    du = Diff(u.ndim - 2, dt)(u)
    return PdeData(t=t, x=x, y=None, z=None, u=u, du=du)

def load_nls(data_path: Path) -> PdeData:
    data = loadmat(data_path)
    t = data['t'].ravel()
    dt = t[1] - t[0]
    x = data['xs'][0][0].ravel()
    u = data['U_exact'][0][0]
    v = data['U_exact'][0][1]
    u = np.stack([u, v], axis=-1)
    du = Diff(u.ndim - 2, dt)(u)
    return PdeData(t=t, x=x, y=None, z=None, u=u, du=du)

def load_advection_diffusion_2d(data_path: Path) -> PdeData:
    '''Source: https://github.com/PhIMaL/DeePyMoD'''
    data = loadmat(data_path)
    usol = np.real(data["Expression1"]).astype("float32")
    usol = usol.reshape((51, 51, 61, 4))
    x = np.array(sorted(set(usol[:, :, :, 0].flatten())))
    y = np.array(sorted(set(usol[:, :, :, 1].flatten())))
    t = np.array(sorted(set(usol[:, :, :, 2].flatten())))
    u = usol[:, :, :, 3:]

    dt = t[1] - t[0]
    du = Diff(u.ndim - 2, dt)(u)
    return PdeData(t=t, x=x, y=y, z=None, u=u, du=du)

def load_reaction_diffusion_2d(data_path: Path) -> PdeData:
    data = np.load(data_path)
    x = data['x']
    y = data['y']
    t = data['t']
    u = data['u'] # (n_x, n_y, n_t, 2)
    dt = t[1] - t[0]
    du = Diff(u.ndim - 2, dt)(u)
    return PdeData(t=t, x=x, y=y, z=None, u=u, du=du)

def load_heat_soil(data_path: Path, trim=True) -> PdeData:
    data = np.load(data_path)
    t = data['t']
    dt = t[1] - t[0]
    u = data['u']
    x = data.get('x', None)
    y = data.get('y', None)
    z = data.get('z', None)
    du = Diff(u.ndim - 2, dt)(u)
    if z is not None and trim:
        t_max = 20
        t = t[:t_max]
        u = u[..., :t_max, :]
        du = du[..., :t_max, :]
    return PdeData(t=t, x=x, y=y, z=z, u=u, du=du)

def load_navier_stokes_channel(data_path: Path, trim=True) -> PdeData:
    data = np.load(data_path)
    t = data['t']
    dt = t[1] - t[0]

    x = data['x']
    y = data['y']
    u = data['u']
    u = u[..., :2] # Remove the pressure component
    du = Diff(u.ndim - 2, dt)(u)
    if trim:
        t_begin = 10
        t_end = 60
        indice = np.s_[4:13, 4:13]
        x = x[indice[0]]
        y = y[indice[1]]
        t = t[t_begin:t_end]
        u = u[indice][..., t_begin:t_end, :]
        du = du[indice][..., t_begin:t_end, :]
    return PdeData(t=t, x=x, y=y, z=None, u=u, du=du)

def load_heat_laser(data_path: Path, trim=True) -> PdeData:
    data = np.load(data_path)

    t = data['t']
    x = data['x']
    y = data['y']
    z = data['z']
    u = data['u']

    dt = t[1] - t[0]
    du = Diff(u.ndim - 2, dt)(u)

    if trim:
        t_max = 20 # Keep only the first 20 time steps
        t = t[:t_max]
        u = u[..., :t_max, :]
        du = du[..., :t_max, :]
    return PdeData(t=t, x=x, y=y, z=z, u=u, du=du)

def load_navier_stokes_cylinder(data_path: Path, trim=True) -> PdeData:
    data_path = Path(data_path)
    u_raw = np.genfromtxt(data_path / 'ns_cylinder_structured_fenics_velocity_and_pressure.csv', delimiter=',')
    grid = np.genfromtxt(data_path / 'ns_cylinder_structured_fenics_mesh.csv', delimiter=',')
    t = np.genfromtxt(data_path / 'ns_cylinder_structured_fenics_timesteps.csv', delimiter=',')

    x_coords = np.array(sorted(set(grid[:, 0])))
    y_coords = np.array(sorted(set(grid[:, 1])))
    u = u_raw.reshape(len(t), len(x_coords), len(y_coords), 3)
    u = np.transpose(u, (1, 2, 0, 3))
    dt = t[1] - t[0]
    du = Diff(u.ndim - 2, dt)(u)

    if trim:
        t_begin = 999
        t_end = 2000
        t_step = 20

        margin_x = int(len(x_coords) * 0.2)
        margin_y = int(len(y_coords) * 0.2)
        subgrid_index = np.s_[margin_x:-2*margin_x, margin_y:-margin_y, t_begin:t_end:t_step]
        u = u[subgrid_index]
        du = du[subgrid_index]
        x_coords = x_coords[subgrid_index[0]]
        y_coords = y_coords[subgrid_index[1]]
        t = t[subgrid_index[2]]
    return PdeData(t=t, x=x_coords, y=y_coords, z=None, u=u, du=du)

def load_reaction_diffusion_cylinder(data_path: Path, trim=True) -> PdeData:
    data_path = Path(data_path)
    u_concentration_raw = np.genfromtxt(data_path / 'diffusion_reaction_cylinder_structured_fenics_concentrations.csv', delimiter=',')
    grid = np.genfromtxt(data_path / 'diffusion_reaction_cylinder_structured_fenics_mesh.csv', delimiter=',')
    t = np.genfromtxt(data_path / 'diffusion_reaction_cylinder_structured_fenics_timesteps.csv', delimiter=',')

    x_coords = np.array(sorted(set(grid[:, 0])))
    y_coords = np.array(sorted(set(grid[:, 1])))
    u = u_concentration_raw.reshape(len(t), len(x_coords), len(y_coords), 6)
    u = np.transpose(u, (1, 2, 0, 3))
    dt = t[1] - t[0]
    du = Diff(u.ndim - 2, dt)(u)

    if trim:
        t_begin = 99
        t_end = 200
        t_step = 2

        margin_x = int(len(x_coords) * 0.2)
        margin_y = int(len(y_coords) * 0.2)
        subgrid_index = np.s_[margin_x:-2*margin_x, margin_y:-margin_y, t_begin:t_end:t_step]
        u = u[subgrid_index]
        du = du[subgrid_index]
        x_coords = x_coords[subgrid_index[0]]
        y_coords = y_coords[subgrid_index[1]]
        t = t[subgrid_index[2]]
    return PdeData(t=t, x=x_coords, y=y_coords, z=None, u=u, du=du)

def add_noise(data: PdeData, snr: float) -> PdeData:
    sigma2 = 10**(-snr/10)
    noise = np.random.randn(*data.u.shape) * np.sqrt(sigma2)
    u = (1 + noise) * data.u
    return PdeData(t=data.t, x=data.x, y=data.y, z=data.z, u=u, du=data.du)

def save_data(data: PdeData, output_path: Path):
    data = data.__dict__
    data = {k: v for k, v in data.items() if v is not None}
    for v in data.values():
        assert isinstance(v, np.ndarray)
    assert data['u'].shape == data['du'].shape
    assert data['t'].shape[0] == data['u'].shape[-2]
    if 'z' in data: # 3D data
        assert data['u'].ndim == 5
        assert data['x'].shape[0] == data['u'].shape[0]
        assert data['y'].shape[0] == data['u'].shape[1]
        assert data['z'].shape[0] == data['u'].shape[2]
    elif 'y' in data: # 2D data
        assert data['u'].ndim == 4
        assert data['x'].shape[0] == data['u'].shape[0]
        assert data['y'].shape[0] == data['u'].shape[1]
    else:
        assert data['u'].ndim == 3
        assert data['x'].shape[0] == data['u'].shape[0]
    assert data['t'].shape[0] == data['u'].shape[-2]
    np.savez(output_path, **data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()

    config = [
        # (original file name, new file name without extension, function to load data)
        ('burgers.mat', 'burgers', load_burgers),
        ('kdv.mat', 'kdv', load_kdv),
        ('kuramoto_sivishinky.mat', 'kuramoto_sivishinky', load_kuramoto_sivishinky),
        ('advection1d.npz', 'advection1d', load_advection1d),
        ('NLS.mat', 'nls', load_nls),
        ('advection_diffusion_2d.mat', 'advection_diffusion_2d', load_advection_diffusion_2d),
        ('reaction_diffusion_data.npz', 'reaction_diffusion_2d', load_reaction_diffusion_2d),
        ('heat_soil_uniform_1d_p1.npz', 'heat_soil_uniform_1d_p1', load_heat_soil),
        ('heat_soil_uniform_2d_p1.npz', 'heat_soil_uniform_2d_p1', load_heat_soil),
        ('heat_soil_uniform_3d_p1.npz', 'heat_soil_uniform_3d_p1', load_heat_soil),
        ('navier_stokes_channel_1.npz', 'navier_stokes_channel', load_navier_stokes_channel),
        ('', 'navier_stokes_cylinder', load_navier_stokes_cylinder),
        ('heat3d_laser.npz', 'heat_laser', load_heat_laser),
        ('', 'reaction_diffusion_cylinder', load_reaction_diffusion_cylinder),
    ]

    snrs = [40, 30, 20, 10]

    for data_path, new_name, load_func in tqdm(config):
        input_path = str(args.input_dir / data_path)
        data = load_func(input_path)
        output_path = args.output_dir / f'{new_name}'
        output_path.mkdir(parents=True, exist_ok=True)
        save_data(data, output_path / f'{new_name}.npz')
        for snr in snrs:
            data_with_noise = add_noise(data, snr)
            save_data(data_with_noise, output_path / f'{new_name}_snr_{snr}.npz')

    # save full data for the trimmed datasets
    config_full = [
        # (original file name, new file name without extension, function to load data)
        ('heat_soil_uniform_3d_p1.npz', 'heat_soil_uniform_3d', load_heat_soil),
        ('navier_stokes_channel_1.npz', 'navier_stokes_channel', load_navier_stokes_channel),
        ('', 'navier_stokes_cylinder', load_navier_stokes_cylinder),
        ('heat3d_laser.npz', 'heat_laser', load_heat_laser),
        ('', 'reaction_diffusion_cylinder', load_reaction_diffusion_cylinder),
    ]
    for data_path, new_name, load_func in tqdm(config_full):
        input_path = str(args.input_dir / data_path)
        data = load_func(input_path, trim=False)
        output_path = args.output_dir / 'full' / f'{new_name}'
        output_path.mkdir(parents=True, exist_ok=True)
        save_data(data, output_path / f'{new_name}.npz')
