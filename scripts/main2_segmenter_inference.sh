#!/bin/bash
#Lucas Sancéré -


################ PARSE CONFIG ARGS

# Extract all needed parameters from mmsegmentation config

config_path=../configs/models/scc_segmenter.yml

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



############### DOWNSAMPLING

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


# Change input dir to the downsampled images now if the downsampling is needed:
if [ $downsample_needed = true ]; then

	input_dir="$input_dir/downsampling/"
fi
# Name of folder "downsmapling" is based on the name chosen on
# src.histo_miner.utils.image_processing.downsample_image_segmenter function arg*

echo "Run mmsegmentation submodule inference..."



############### UPDATE CONFIG

# Rewrite segmenter config to take into account mmsegementation.yml config
# cfg-option from mmsegmentation is not working to overwritte config (with this version)

FILE_PATH=./configs/_base_/datasets/mc_sccskin184.py

LINE_INFERENCE_ROOT="4"
TEXT_INFERENCE_ROOT="inference_root='$input_dir'"
LINE_SAMPLES_PER_GPU="54"
TEXT_SAMPLES_PER_GPU="    samples_per_gpu=$samples_per_gpu,"
LINE_WORKERS_PER_GPU="55"
TEXT_WORKERS_PER_GPU="    workers_per_gpu=$workers_per_gpu,"

# Check if the file exists
if [ ! -f "$FILE_PATH" ]; then
  echo "File not found!"
fi

# Create a temporary file
TEMP_FILE=$(mktemp)

# Debug: Display the variables
echo "Replacing lines in $FILE_PATH"
echo "Line $LINE_INFERENCE_ROOT with: $TEXT_INFERENCE_ROOT"
echo "Line $LINE_SAMPLES_PER_GPU with: $TEXT_SAMPLES_PER_GPU"
echo "Line $LINE_WORKERS_PER_GPU with: $TEXT_WORKERS_PER_GPU"

# Replace the specified lines with the new text
awk -v line1="$LINE_INFERENCE_ROOT" -v text1="$TEXT_INFERENCE_ROOT" \
    -v line2="$LINE_SAMPLES_PER_GPU" -v text2="$TEXT_SAMPLES_PER_GPU" \
    -v line3="$LINE_WORKERS_PER_GPU" -v text3="$TEXT_WORKERS_PER_GPU" \
'NR == line1 {print text1; next}
 NR == line2 {print text2; next}
 NR == line3 {print text3; next}
 {print $0}' "$FILE_PATH" > "$TEMP_FILE"

# Check if the temporary file was created successfully
if [ $? -ne 0 ]; then
  echo "Error creating the temporary file."
  exit 1
fi

# Overwrite the original file with the temporary file
mv "$TEMP_FILE" "$FILE_PATH"
echo "Config updated"



############### RUN SEGMENTER INFERENCE 


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
