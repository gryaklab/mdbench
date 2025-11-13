#! /bin/bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ "$1" == "--clean" ]; then
    for env in $(conda env list | awk '{print $1}' | grep '^md-bench-'); do
        conda env remove -n "$env" -y
    done
    exit 0
fi

requirements_path=$(realpath requirements.txt)

pushd mdbench
algorithms=$(find algorithms/ -mindepth 2 -maxdepth 2 -type d ! -name '_*')

# Function to check if environment exists
check_env_exists() {
    conda env list | grep -q "^$1 "
    return $?
}

# For each algorithm
for algo in $algorithms; do
    algo_name=$(basename $algo)
    env_name="md-bench-$algo_name"

    if check_env_exists "$env_name"; then
        echo "Environment $env_name already exists, skipping..."
    else
        pushd "$algo"
        echo "Creating environment $env_name..."
        conda env create -f "environment.yml" -n "$env_name" -y
        conda activate "$env_name"
        pip install -r "$requirements_path"
        if [ -f "install.sh" ]; then
            chmod +x install.sh
            bash install.sh
        fi
        conda deactivate
        popd
    fi
done
