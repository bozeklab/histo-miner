#!/bin/bash
#Lucas Sancéré -

## As hovernet is a submodule of the repo, it is better to use sh files and not python script
## as we need activate corresponding env and the inference was also originally performed
## thourgh as sh file (see src/models/hover_net/run_wsi.sh)


################ PARSE CONFIG ARGS

# Extract all needed parameters from hovernet config

config_path=./configs/models/hovernet.yml

yaml() {
    python -c "import yaml;print(yaml.safe_load(open('$1'))$2)"
}

hovernet_mode=$(yaml $config_path "['key1']['key2']['key3']")
gpulist=$(yaml $config_path "['key1']['key2']['key3']") 
checkpoints_dir=$(yaml $config_path "['key1']['key2']['key3']")
input_dir=$(yaml $config_path "['key1']['key2']['key3']")
output_dir=$(yaml $config_path "['key1']['key2']['key3']")
cache_path=$(yaml $config_path "['key1']['key2']['key3']")



############### SCRIPT

conda deactivate
conda activate hovernet_submodule_test1

cd ./src/models/hover_net/

#May need to export the LIBRARY PATH as follow
# export LD_LIBRARY_PATH="/data/user/miniconda3/envs/hovernet_submodule_test1/lib/:$LD_LIBRARY_PATH"


# Download weigths to add

### Add script to download weghts only if there are not already downloaded
### for dev purposes, download it from google drive
### for publication purposes, download if from Zenodo







# Set number of open files limit to 10 000! 
ulimit -n 10000

echo "Run hover_net submodule inference..."

# Hovernet inference parameters 

if [ "$hovernet_mode" == "$wsi" ]; then
    python run_infer.py \
    --gpu= gpulist \
    --nr_types=6 \
    --type_info_path=./type_info_SCC.json \
    --model_path= checkpoints_dir \
    --model_mode=fast \
    --batch_size=64 \
    wsi \
    --input_dir= input_dir \
    --output_dir= output_dir \
    --cache_path= cache_path

if [ "$hovernet_mode" == "$tile" ]; then
    python run_infer.py \
    --gpu= gpulist \
    --nr_types=6 \
    --type_info_path=./type_info_SCC.json \
    --model_path= checkpoints_dir \
    --model_mode=fast \
    --batch_size=64 \
    tile \
    --input_dir= input_dir \
    --output_dir= output_dir \




