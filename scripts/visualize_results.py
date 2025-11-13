from pathlib import Path
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

ode_fig_size = (8, 12)
pde_fig_size = (6, 5)
overall_fig_size = (8, 5)
hue_order = ['Clean', '40', '30', '20', '10']
marker_size = 10

dataset_name_mapping = {
    'burgers': 'Burgers',
    'kdv': 'KdV',
    'kuramoto_sivishinky': 'Kuramoto-Sivishinky',
    'advection1d': 'Advection',
    'nls': 'Nonlinear Schrödinger',
    'advection_diffusion_2d': 'Advection-Diffusion (2D)',
    'reaction_diffusion_2d': 'Reaction-Diffusion (2D)',
    'heat_soil_uniform_1d_p1': 'Heat (1D)',
    'heat_soil_uniform_2d_p1': 'Heat (2D)',
    'heat_soil_uniform_3d_p1': 'Heat (3D)',
    'navier_stokes_channel': 'Navier-Stokes (Channel)',
    'heat_laser': 'Heat (Laser)',
    'navier_stokes_cylinder': 'Navier-Stokes (Cylinder)',
    'reaction_diffusion_cylinder': 'Reaction-Diffusion (Cylinder)',
}

pde_dataset_order = [
    'Advection',
    'Burgers',
    'KdV',
    'Kuramoto-Sivishinky',
    'Advection-Diffusion (2D)',
    'Heat (1D)',
    'Heat (2D)',
    'Heat (3D)',
    'Heat (Laser)',
    'Reaction-Diffusion (2D)',
    'Nonlinear Schrödinger',
    'Navier-Stokes (Channel)',
    'Navier-Stokes (Cylinder)',
    'Reaction-Diffusion (Cylinder)',
]

common_pde_datasets = [
    'Advection',
    'Burgers',
    'KdV',
    'Kuramoto-Sivishinky',
    'Advection-Diffusion (2D)',
    'Heat (1D)',
]

method_name_mapping = {
    'esindy': 'E-SINDy',
    'eql': 'EQL',
    'sindy': 'SINDy',
    'operon': 'Operon',
    'pysr': 'PySR',
    'wsindy': 'WSINDy',
    'ewsindy': 'E-WSINDy',
    'bayesian': 'Bayesian',
    'dso': 'uDSR',
    'odeformer': 'ODEFormer',
    'pdefind': 'PDEFind',
    'deepmod': 'DeepMoD',
    'e2e': 'End2end',
}

pde_method_order = [
    'PDEFind',
    'Bayesian',
    'WSINDy',
    'E-WSINDy',
    'DeepMoD',
    'PySR',
    'Operon',
]

ode_method_order = [
    'SINDy',
    'E-SINDy',
    'PySR',
    'Operon',
    'EQL',
    'ODEFormer',
    'End2end'
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default="result")
    parser.add_argument("--output_dir", type=Path, default="figs")
    return parser.parse_args()

def plot_by_dataset(results: pd.DataFrame, figures_path: Path, data_type: str):
    df = results[results.data_type == data_type]
    for method_name in df.method.unique():
        df_method = df[df.method == method_name]
        plt.figure(figsize=pde_fig_size if data_type == 'pde' else ode_fig_size)
        sns.stripplot(
            data=df_method,
            x='nmse',
            y='dataset',
            hue='snr',
            hue_order=hue_order,
            palette='magma',
            size=marker_size,
            jitter=False,
            marker='X',
            order=pde_dataset_order if data_type == 'pde' else None,
        )

        handles, labels = plt.gca().get_legend_handles_labels()
        ordered = [handles[labels.index(label)] for label in hue_order]
        plt.legend(title="Noise", handles=ordered)

        plt.title(method_name_mapping[method_name])
        plt.xscale('log')
        plt.xlabel('NMSE')
        plt.ylabel('')
        plt.tight_layout()
        plt.savefig(figures_path / f'nmse-{method_name}-{data_type}.pdf')
        plt.close()

        plt.figure(figsize=pde_fig_size if data_type == 'pde' else ode_fig_size)
        # How to control the order of y labels? answer: use order parameter
        sns.stripplot(
            data=df_method,
            x='complexity',
            y='dataset',
            hue='snr',
            hue_order=hue_order,

            palette='magma',
            size=10,
            jitter=False,
            marker='X',
            order=pde_dataset_order if data_type == 'pde' else None,
        )

        handles, labels = plt.gca().get_legend_handles_labels()
        ordered = [handles[labels.index(label)] for label in hue_order]
        plt.legend(title="SNR", handles=ordered)

        plt.title(method_name_mapping[method_name])
        plt.xscale('log')
        plt.ylabel('')
        plt.xlabel('Complexity')
        plt.tight_layout()
        plt.savefig(figures_path / f'complexity-{method_name}-{data_type}.pdf')
        plt.close()

def plot_by_method(results: pd.DataFrame, figures_path: Path, data_type: str):
    df = results[results.data_type == data_type]
    if data_type == 'pde':
        df = df[df.dataset.isin(common_pde_datasets)]
    df.loc[:, 'method'] = df['method'].map(method_name_mapping, na_action='ignore').fillna(df['method'])

    fig =plt.figure(figsize=overall_fig_size)
    ax1 = plt.subplot(1, 2, 1)
    sns.boxplot(
        data=df,
        x='nmse',
        y='method',
        hue='snr',
        hue_order=hue_order,
        palette='magma',
        fill=True,
        gap=0.2,
        order=pde_method_order if data_type == 'pde' else ode_method_order,
    )
    plt.xlabel('NMSE')
    plt.xscale('log')
    plt.ylabel('')

    handles, labels = plt.gca().get_legend_handles_labels()
    ordered = [handles[labels.index(label)] for label in hue_order]

    ax2 = plt.subplot(1, 2, 2)
    sns.boxplot(
        data=df,
        x='complexity',
        y='method',
        hue='snr',
        hue_order=hue_order,
        palette='magma',
        fill=True,
        gap=0.2,
        order=pde_method_order if data_type == 'pde' else ode_method_order,
    )
    plt.xlabel('Complexity')
    plt.ylabel('')
    plt.yticks([])

    ax1.legend().remove()
    ax2.legend().remove()

    handles, labels = ax2.get_legend_handles_labels()
    ordered = [handles[labels.index(label)] for label in hue_order]

    plt.legend(
        title="SNR (dB)",
        loc='upper center',
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(labels),
        frameon=False,
        bbox_transform=fig.transFigure,
        handles=ordered,
    )
    plt.xscale('log')
    # plt.tight_layout()
    plt.savefig(figures_path / f'{data_type}.pdf', bbox_inches='tight')

if __name__ == "__main__":
    args = parse_args()
    results_dir = args.results_dir
    figures_path = args.output_dir
    results_path = list(results_dir.glob(f"*.csv"))[0]
    results = pd.read_csv(results_path)

    # Remove deepmod results because it is not implemented fully yet
    # results = results[~(results.method == 'deepmod')]

    # Remove eql results for PDEs because most of them are failed
    results = results[~((results.data_type == 'pde') & (results.method == 'eql'))]

    results = results.dropna(subset=['nmse', 'complexity'])
    # Make a warning if there are duplicate rows
    if results.duplicated(subset=['method', 'dataset', 'snr']).any():
        print("""Warning: There are duplicate rows in the combined results.
        Removing the dupliacets for the plots.""")
        results = results.drop_duplicates(subset=['method', 'dataset', 'snr'])

    results['dataset'] = results['dataset'].map(dataset_name_mapping, na_action='ignore').fillna(results['dataset'])
    figures_path.mkdir(exist_ok=True)
    for data_type in ['pde', 'ode']:
        plot_by_method(results, figures_path, data_type)
        plot_by_dataset(results, figures_path, data_type)

