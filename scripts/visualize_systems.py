import argparse
from pathlib import Path
import math

import tqdm
import numpy as np
from findiff import Diff
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from mdbench.data_loader import (
    load_pde_datasets,
    PdeDataset,
)

def visualize_pde(dataset: PdeDataset, save_to : Path):
    t = dataset.t
    s = dataset.s
    u = dataset.u

    # sample 1000 frames from the dataset
    total_samples = 100
    frames = range(0, min(len(t), total_samples), math.ceil(len(t) / total_samples))

    dt = t[1] - t[0]
    u_t = Diff(u.ndim-2, dt)(u)

    spatial_dim = u.ndim - 2
    dim = u.shape[-1]
    fig, ax = plt.subplots(1, dim, figsize=(3*dim, 3))
    if not isinstance(ax, np.ndarray):
        ax = [ax]

    if spatial_dim == 1:
        x = s
        lines = [ax[d].plot(s, u[..., 0, d])[0] for d in range(dim)]
        def update(frame):
            for d in range(dim):
                lines[d].set_ydata(u[..., frame, d])
                ax[d].set_xlabel('x')
            plt.suptitle(f"t = {t[frame]:.2f}s")
            plt.tight_layout()
            return lines,
    if spatial_dim == 2:
        min_x = s[:, 0].min()
        max_x = s[:, 0].max()
        min_y = s[:, 1].min()
        max_y = s[:, 1].max()
        cax = [ax[d].imshow(u[..., 0, d].T, extent=[min_y, max_y, min_x, max_x], origin='lower', cmap='viridis', vmin=u.min(), vmax=u.max()) for d in range(dim)]
        for d in range(dim):
            fig.colorbar(cax[d], ax=ax[d])

        def update(frame):
            for d in range(dim):
                cax[d].set_array(u[..., frame, d].T)
                ax[d].set_xlabel('x')
                ax[d].set_ylabel('y')
                plt.suptitle(f"t = {t[frame]:.2f}s")
            plt.tight_layout()
            return cax,

    name = dataset.name.lower().replace(' ', '_')
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50)
    if not save_to.exists():
        save_to.mkdir(parents=True)
    path = save_to / f"{name}.gif"
    ani.save(path, writer='pillow', dpi=75)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the visualizations')
    args = parser.parse_args()

    for dataset in tqdm.tqdm(load_pde_datasets()):
        visualize_pde(dataset, save_to=Path(args.output_dir))