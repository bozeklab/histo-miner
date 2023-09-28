#Lucas Sancéré -

################ RMQ

####### Add the parameters in a config files and read it HERE on the sh scripts
####### Except the thing that shouldn't be changed like type info path and model mode?

#!/bin/bash

# Source the configuration file
. config.txt

# Access configuration values
echo "API_KEY: $API_KEY"
echo "DATABASE_URL: $DATABASE_URL"

if [ "$DEBUG" = "true" ]; then
    echo "Debug mode is enabled."
else
    echo "Debug mode is disabled."
fi


############### SCRIPT

conda deactivate
conda activate hovernet_submodule_test1

cd ./src/models/hover_net/

#May need to export the LIBRARY PATH as follow
# export LD_LIBRARY_PATH="/data/user/miniconda3/envs/hovernet_submodule_test1/lib/:$LD_LIBRARY_PATH"


# Download weigths to add



# Set number of open files limit to 10 000! 
ulimit -n 10000

echo "Run hover_net submodule inference..."

# Hovernet inference parameters 
python run_infer.py \
--gpu='1' \
--nr_types=6 \
--type_info_path=./type_info_SCC.json \
--model_path=tocomplete \
--model_mode=fast \
--batch_size=64 \
wsi \
--input_dir=/path/to/data/input/to/add/ \
--output_dir=//path/to/data/output/to/add/ \
--cache_path=/path/to/cache/to/add/


# Example of tile mode:
# python run_infer.py \
# --gpu='1' \
# --nr_types=6 \
# --type_info_path=./type_info_SCC.json \
# --model_path= \
# --model_mode=fast \
# --batch_size=64 \
# wsi \
# --input_dir=/path/to/add/ \
# --output_dir=/path/to/add/ 


