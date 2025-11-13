#! /bin/bash
eval "$(conda shell.bash hook)"

set -e

USAGE="Usage: ./run.sh --algorithm <algorithm> --data_type <data_type>
    [--result_dir <result_dir>] [--with_noise]
    [--dataset <dataset>] [--dataset_path <dataset_path>] [--save_pareto] [--n_jobs <n_jobs>]
    [--timeout <timeout>]
  --algorithm: algorithm name
  --data_type: data type. Can be 'pde' or 'ode'
  --result_dir: result directory. If not specified, the path is set to 'result'.
  --with_nosie: whether to use noisy datasets. If not specified, only clean datasets will be used.
  --dataset: dataset name. If not specified, all datasets will be used.
  --dataset_path: dataset path. If not specified, the path is set to 'data'.
  --save_pareto: whether to save the pareto front. If not specified (default), the pareto front will not be saved.
  --n_jobs: number of jobs to run in parallel. Default is -1, which means all the CPU cores will be used.
  --timeout: timeout in seconds. Default is 12*60*60 (12 hours).

Examples:
  ./run.sh --algorithm pdefind --data_type pde --result_dir results
  ./run.sh --algorithm pysr --data_type ode --result_dir results --with_noise
  ./run.sh --algorithm pysr --data_type pde --result_dir results --with_noise
"

if [ "$1" == '-h' ] || [ "$1" == '--help' ] || [ "$1" == 'help' ]; then
    echo "$USAGE"
    exit 0
fi

OPTIONS=""
LONGOPTS=result_dir:,algorithm:,dataset:,dataset_path:,data_type:,with_noise,save_pareto,n_jobs:,timeout:
PARSED=$(getopt --options="$OPTIONS" --longoptions="$LONGOPTS" --name "$0" -- "$@")
if [[ $? -ne 0 ]]; then
  exit 2
fi

eval set -- "$PARSED"

# Default values
RESULT_DIR="result"
DATASET="all"
DATASET_PATH="data" # it should include `pde` and `ode` directories
WITH_NOISE=false
SAVE_PARETO=""
N_JOBS=-1
TIMEOUT=43200  # 12*60*60

while true; do
  case "$1" in
    --result_dir)
      RESULT_DIR="$2"
      shift 2
      ;;
    --algorithm)
      ALGORITHM="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --dataset_path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --data_type)
      DATA_TYPE="$2"
      shift 2
      ;;
    --with_noise)
      WITH_NOISE=true
      shift
      ;;
    --save_pareto)
      SAVE_PARETO="--save_pareto"
      shift
      ;;
    --n_jobs)
      N_JOBS="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Invalid option: $1"
      echo "$USAGE"
      exit 3
      ;;
  esac
done

if [ -z "$ALGORITHM" ]; then
    echo "Error: algorithm is required"
    exit 1
fi

if [ -z "$DATA_TYPE" ]; then
    echo "Error: data type is required"
    exit 1
fi

ALGORITHM_PATH=$(find mdbench/algorithms/ -mindepth 2 -maxdepth 2 -type d ! -name '_*' -name $ALGORITHM)
if [ -z "$ALGORITHM_PATH" ]; then
    echo "Error: algorithm $ALGORITHM not found"
    echo "Available algorithms:"
    ALL_ALGORITHMS=$(find mdbench/algorithms/ -mindepth 2 -maxdepth 2 -type d ! -name '_*')
    for algorithm in $ALL_ALGORITHMS; do
        echo "  $(basename $algorithm)"
    done
    exit 1
fi

ALGORITHM_TYPE=$(basename $(dirname $ALGORITHM_PATH))

if [ -z "$ALGORITHM_PATH" ]; then
    echo "Error: algorithm $ALGORITHM not found"
    exit 1
fi

ALL_DATASETS=$(find $DATASET_PATH/$DATA_TYPE -name "*.npz")
if [ $DATASET == "all" ]; then
    DATASET=$ALL_DATASETS
else
    DATASET=$(echo "$ALL_DATASETS" | grep "$DATASET")
fi
if [ "$WITH_NOISE" == false ]; then
    DATASET=$(echo "$DATASET" | grep -v "snr")
fi

echo "Running $ALGORITHM on $DATA_TYPE datasets:"

n=1
for dataset in $DATASET; do
    echo " $n. $(basename $dataset)"
    n=$((n+1))
done

echo "Activating environment md-bench-$ALGORITHM"
conda activate "md-bench-$ALGORITHM"

python -m mdbench.evaluate_method \
        --algorithm $ALGORITHM \
        --algorithm_type $ALGORITHM_TYPE \
        --data_type $DATA_TYPE \
        --result_dir $RESULT_DIR \
        --datasets $DATASET \
        --n_jobs $N_JOBS \
        --timeout $TIMEOUT \
        $SAVE_PARETO