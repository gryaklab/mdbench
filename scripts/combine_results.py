from pathlib import Path
import json
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=Path, required=True, default="result")
    parser.add_argument("--output_dir", type=Path, required=False)
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = args.result_dir
    return args


def parse_result_file(result_file: Path):
    results = [json.loads(line) for line in open(result_file).readlines()]
    results = [(x['nmse'], x['complexity'], x['dataset'], x['equation'], x.get('hyperparameters', '')) for x in results]
    results = pd.DataFrame(results, columns=['nmse', 'complexity', 'dataset', 'equation', 'hyperparameters'])

    results.equation = results.equation.str.replace('\n', ' | ')

    noisy_rows = results.dataset.str.contains('snr')
    results['snr'] = 'Clean'
    results.loc[noisy_rows, 'snr'] = results.dataset[noisy_rows].str.split('_').str[-1]
    results.dataset = results.dataset.str.split('_snr').str[0]

    cols = results.columns.tolist()
    equation_col_index = cols.index('equation')
    cols[-1], cols[equation_col_index] = cols[equation_col_index], cols[-1]
    results = results[cols]
    return results

if __name__ == "__main__":
    args = parse_args()
    result_dir = args.result_dir
    output_path = args.output_dir / "combined.csv"

    combined_results = []
    data_types = ['pde', 'ode']
    for data_type in data_types:
        results_paths = list(result_dir.glob(f"*{data_type}.jsonl"))

        def get_method_name(path: Path):
            file_name = path.name
            method_name = ''.join(file_name.split('-')[:-1])
            return method_name
        results = {get_method_name(p): parse_result_file(p) for p in results_paths}
        for method_name, result in results.items():
            result['method'] = method_name
        results = pd.concat(results.values())
        results['data_type'] = data_type
        combined_results.append(results)
    combined_results = pd.concat(combined_results)
    combined_results.to_csv(output_path, index=False)
