import os
import importlib
import argparse
import json
from pathlib import Path
from itertools import product
import time
from dataclasses import dataclass
import signal

import numpy as np
from joblib import Parallel, delayed

from mdbench import metrics
from mdbench import data_loader
from mdbench.utils import is_excluded_dataset

np.random.seed(42)

@dataclass
class Result:
    equation: str = ''
    score: float = -1
    nmse: float = None
    complexity: int = None
    kwargs: dict = None
    train_time: float = None
    dataset: str = None
    error: str = ''
    pareto_front: list = None

    def to_dict(self):
        return {
            'dataset': self.dataset,
            'nmse': self.nmse,
            'complexity': self.complexity,
            'train_time': self.train_time,
            'score': self.score,
            'error': self.error,
            'hyperparameters': self.kwargs,
            'equation': self.equation,
            'pareto_front': self.pareto_front,
        }

def handler(signum, frame):
    raise TimeoutError('Timeout')
signal.signal(signal.SIGALRM, handler)

def set_env_vars(n_jobs):
    if n_jobs == -1:
        return
    n_jobs = str(n_jobs)
    os.environ['OMP_NUM_THREADS'] = n_jobs
    os.environ['OPENBLAS_NUM_THREADS'] = n_jobs
    os.environ['MKL_NUM_THREADS'] = n_jobs

def save_result(result: Result, path: Path):
    with open(path, 'a') as f:
        f.write(json.dumps(result.to_dict()) + "\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--algorithm_type", type=str, required=True, choices=['sr', 'ode', 'pde'])
    parser.add_argument("--data_type", type=str, required=True, choices=['ode', 'pde'])
    parser.add_argument("--data_path", type=Path, required=False,
                        help="path to the data file, only required for ode")
    parser.add_argument("--result_dir", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True, nargs='+')
    parser.add_argument("--save_pareto", action='store_true', required=False, default=False)
    parser.add_argument("--n_jobs", type=int, required=False, default=-1)
    parser.add_argument("--timeout", type=int, required=False, default=12*60*60)
    return parser.parse_args()

def hyperparameter_pareto_front(model_cls, time_train, observation_train, derivative_train, hyperparameters, s=None, n_jobs=None):
    '''Find the best hyperparameters for the model

    Splits the data into training and validation sets (80:20)
    and finds the best hyperparameters by maximizing the validation fitness.

    Fitness score is defined as a combination of the NMSE and the complexity of the model:
    `score = 1/(1 + nmse) + lam*np.exp(-complexity/L)`

    Args:
        model_cls: The model class to use
        time_train: The time data for training
        observation_train: The observation trajectory for training
        derivative_train: The derivative (target) for training
        hyperparameters: The hyperparameters to search over
        s: The spatial grid, only required for pde
        n_jobs: The number of jobs to run in parallel
    '''

    validation_cutoff = int(len(time_train) * 0.2)
    time_train = time_train[:-validation_cutoff]
    time_val = time_train[-validation_cutoff:]
    observation_train = observation_train[..., :-validation_cutoff, :]
    observation_val = observation_train[..., -validation_cutoff:, :]
    derivative_train = derivative_train[..., :-validation_cutoff, :]
    derivative_val = derivative_train[..., -validation_cutoff:, :]

    prod = list(product(*hyperparameters.values()))
    hp_candidates = [dict(zip(hyperparameters.keys(), hp)) for hp in prod]
    scores = []
    def evaluate_single_hyperparameter(kwargs):
        model = model_cls(**kwargs)
        if s is not None:
            model.set_spatial_grid(s)
        try:
            model.fit(time_train, observation_train, derivative_train)
            complexity = model.complexity()
        except Exception as e:
            print('Warning: Failed to train the model for hyperparameters: ', kwargs)
            print(e)
            return None

        y_pred = model.predict(time_val, observation_val)
        nmse = metrics.nmse(derivative_val, y_pred)
        score = metrics.fitness(nmse, complexity)
        result = Result(
            equation=str(model.to_str()),
            score=float(score),
            nmse=float(nmse),
            complexity=int(complexity),
            kwargs=kwargs,
        )
        return result

    all_results = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(delayed(evaluate_single_hyperparameter)(kwargs) for kwargs in hp_candidates)
    all_results = [r for r in all_results if r is not None]
    complexities = [r.complexity for r in all_results if r is not None]
    complexities = np.array(sorted(set(complexities)))
    pareto_front = []
    for complexity in complexities:
        results = [r for r in all_results if r is not None and r.complexity == complexity]
        best_result = min(results, key=lambda x: x.nmse)
        pareto_front.append(best_result)
    return pareto_front

def train_and_evaluate(model_cls, dataset, hyperparameters=None, common_kwargs=None, save_pareto=False):
    result = {}
    spatial_gird = dataset.get_s() if isinstance(dataset, data_loader.PdeDataset) else None
    t_train, u_train, du_train, du_true_train = dataset.get_train()
    t_test, u_test, du_test, du_true_test = dataset.get_test()

    t_train_start = time.time()
    if hyperparameters:
        pareto_front = hyperparameter_pareto_front(
            model_cls=model_cls,
            time_train=t_train,
            observation_train=u_train,
            derivative_train=du_train,
            hyperparameters=hyperparameters,
            s=spatial_gird,
            n_jobs=common_kwargs['n_jobs'],
        )
        best_hyperparameters = max(pareto_front, key=lambda x: x.score).kwargs
    else:
        best_hyperparameters = {}
    model = model_cls(**best_hyperparameters, **common_kwargs)
    if spatial_gird is not None:
        model.set_spatial_grid(spatial_gird)
    # Train the model on all the training data
    model.fit(t_train, u_train, du_train)
    t_train_end = time.time()

    y_pred = model.predict(t_test, u_test)
    y_pred = y_pred.reshape(du_true_test.shape)
    nmse = float(metrics.nmse(du_true_test, y_pred))
    equation = model.to_str()
    try:
        complexity = int(model.complexity())
    except Exception as e:
        print('Warning: Failed to compute complexity for the equation: ', equation)
        print(e)
        complexity = np.nan
    score = metrics.fitness(nmse, complexity)
    result = Result(
        equation=equation,
        nmse=nmse,
        complexity=complexity,
        score=score,
        train_time=t_train_end - t_train_start,
        dataset=dataset.name,
    )
    if hyperparameters:
        result.kwargs = str(best_hyperparameters)
        if save_pareto:
            result.pareto_front = str(pareto_front)
    return result

def train_and_save(estimator_cls, dataset, hyper_params, common_kwargs, save_pareto, result_path):
    try:
        result = train_and_evaluate(estimator_cls, dataset, hyper_params, common_kwargs, save_pareto)
        save_result(result, result_path)
    except Exception as e:
        print(f"Error in {dataset.name}: {e}")
        save_result(Result(error=str(e), dataset=dataset.name), result_path)


if __name__ == "__main__":
    args = parse_args()
    algorithm_name = args.algorithm
    algorithm_type = args.algorithm_type
    data_type = args.data_type
    timeout = args.timeout

    set_env_vars(args.n_jobs)

    result_path = args.result_dir / f"{algorithm_name}-{data_type}.jsonl"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module(f'mdbench.algorithms.{algorithm_type}.{algorithm_name}.regressor')
    estimator_cls = getattr(module, 'Estimator')
    hyper_params = getattr(module, 'hyper_params', None)
    datasets = data_loader.load_datasets(args.datasets, algorithm_type, data_type)
    for dataset in datasets:
        dataset_name = dataset.name if 'snr' not in dataset.name else dataset.name.split('_snr_')[0]
        if is_excluded_dataset(dataset_name, algorithm_name):
            print(f"Skipping {algorithm_name} on {dataset.name}")
            continue
        print(f"Running {algorithm_name} on {dataset.name}")
        common_kwargs = {
            'n_jobs': args.n_jobs,
            'symbol_names': getattr(dataset, 'symbol_names', None),
            'target_names': getattr(dataset, 'target_names', None),
        }

        try:
            signal.alarm(timeout)
            try:
                train_and_save(estimator_cls, dataset, hyper_params, common_kwargs, args.save_pareto, result_path)
            finally:
                signal.alarm(0)
        except TimeoutError:
            save_result(Result(dataset=dataset_name, error='Timeout'), result_path)