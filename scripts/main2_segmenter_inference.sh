#!/bin/bash
#Lucas Sancéré -


################ PARSE CONFIG ARGS

# Extract all needed parameters from mmsegmentation config

config_path=../configs/models/mmsegmentation.yml

# We need yaml lib so we reactivate histo-miner env if it was not done befre
conda deactivate
conda activate histo-miner-env-2

# We extract all parameters now:
yaml() {
    python -c "import yaml;print(yaml.safe_load(open('$1'))$2)"
}


downsample_needed=$(yaml $config_path "['inference']['downsample_needed']") 
gpu=$(yaml $config_path "['inference']['gpu']") 
samples_per_gpu=$(yaml $config_path "['inference']['samples_per_gpu']")
workers_per_gpu=$(yaml $config_path "['inference']['workers_per_gpu']")    
checkpoints=$(yaml $config_path "['inference']['paths']['checkpoints']")
input_dir=$(yaml $config_path "['inference']['paths']['input_dir']")
output_dir=$(yaml $config_path "['inference']['paths']['output_dir']")

# run_on_slurm=$(yaml $config_path "['inference']['run_on_slurm']")
# slurm_partition=$(yaml $config_path "['inference']['slurm_param']['partition']")
# slurm_job_name=$(yaml $config_path "['inference']['slurm_param']['job_name']")



############### SCRIPT

if [ $downsample_needed = true ]; then

	echo "Downsample images..."

	python -c "import sys; sys.path.append('../'); \
	from src.histo_miner.utils.image_processing import downsample_image_segmenter; \
	downsample_image_segmenter('$input_dir')"

fi


if [ $downsample_needed = false ]; then

	echo "Warning: downsampling of images is skipped."
fi



conda deactivate
conda activate mmsegmentation2

cd ../src/models/mmsegmentation/


# May need to export the LIBRARY PATH as follow
export LD_LIBRARY_PATH="/data/lsancere/miniconda3/envs/mmsegmentation2/$LD_LIBRARY_PATH"


# --------------------------------------------------------
# TO DO? WE CAN ALSO DOWNLOAD USING THE MAIN README
# No need Yet----
# Download weigths to add
## Add script to download weights only if there are not already downloaded
## for dev purposes, download it from google drive
## for publication purposes, download if from Zenodo
# --------------------------------------------------------


# Change input dir to the downsampled images now if the downsampling is needed:
if [ $downsample_needed = true ]; then

	input_dir="$input_dir/downsampling/"
fi
# Name of folder "downsmapling" is based on the name chosen on
# src.histo_miner.utils.image_processing.downsample_image_segmenter function arg*

echo "Run mmsegmentation submodule inference..."


# FOR DEV PURPOSE ---------------------------------------------------------------------------
# Create code to overwritte config in the following way;
# - erase the line with inference_root and rewrite it with inference_root=$input_dir
# - erase the line with data.samples_per_gpu and rewrite it with data.samples_per_gpu=$samples_per_gpu
# - erase the line with data.workers_per_gpu=$workers_per_gpu and rewrite it with data.workers_per_gpu=$workers_per_gpu=$input_dir

# explain that cfg-option is not working with our version of mmsegmentation very likely


if [ $gpu <= 1 ]; then
	python ./tools/test.py \
	configs/segmenter/segmenter_vit-l_SCCJohannes141_mask_8x1_640x640_160k_ade20k.py \ 
    $checkpoints \ 
	--format-only \
	--eval-options "imgfile_prefix=$output_dir" 
fi


if [ $gpu > 1 ]; then
	./tools/dist_test.sh \
	configs/segmenter/segmenter_vit-l_SCCJohannes141_mask_8x1_640x640_160k_ade20k.py \
    $checkpoints \
    $gpu \
    --format-only \
	--eval-options  "imgfile_prefix=$output_dir" 
fi


# return to previous path and reeactivate histo-miner env for following steps
conda deactivate
conda activate histo-miner-env-2
cd "$OLDPWD"
