#!/bin/bash
#Lucas Sancéré -

## As hovernet is a submodule of the repo, it is better to use sh files and not python script
## as we need activate corresponding env and the inference was also originally performed
## thourgh as sh file (see src/models/hover_net/run_wsi.sh)


################ PARSE CONFIG ARGS

# Extract all needed parameters from hovernet config

config_path=../configs/models/scc_hovernet.yml

# We need yaml lib so we reactivate histo-miner env if it was not done befre
conda deactivate
conda activate histo-miner-env

# We extract all parameters now:
yaml() {
    python -c "import yaml;print(yaml.safe_load(open('$1'))$2)"
}

hovernet_mode=$(yaml $config_path "['inference']['hovernet_mode']")
gpulist=$(yaml $config_path "['inference']['gpulist']") 
checkpoints=$(yaml $config_path "['inference']['paths']['checkpoints']")
input_dir=$(yaml $config_path "['inference']['paths']['input_dir']")
output_dir=$(yaml $config_path "['inference']['paths']['output_dir']")
cache_path=$(yaml $config_path "['inference']['paths']['cache_path']")



############### SCRIPT

conda deactivate
conda activate hovernet_submodule

cd ../src/models/hover_net/


#May need to export the LIBRARY PATH as follow
# export LD_LIBRARY_PATH="/data/lsancere/miniconda3/envs/hovernet_submodule_test1/lib/:$LD_LIBRARY_PATH"


# Set number of open files limit to 10 000! 
ulimit -n 100000

echo "Run hover_net submodule inference..."

# Hovernet inference parameters 

if [ "$hovernet_mode" == "wsi" ]; then
    python run_infer.py \
    --gpu=$gpulist \
    --nr_types=6 \
    --type_info_path=./type_info_SCC.json \
    --model_path=$checkpoints \
    --model_mode=fast \
    --batch_size=64 \
    wsi \
    --input_dir=$input_dir \
    --output_dir=$output_dir \
    --cache_path=$cache_path 
fi


if [ "$hovernet_mode" == "tile" ]; then
    python run_infer.py \
    --gpu=$gpulist \
    --nr_types=6 \
    --type_info_path=./type_info_SCC.json \
    --model_path=$checkpoints \
    --model_mode=fast \
    --batch_size=64 \
    tile \
    --input_dir=$input_dir \
    --output_dir=$output_dir
fi


# return to previous path and reeactivate histo-miner env for following steps
conda deactivate
conda activate histo-miner-env
cd "$OLDPWD"
