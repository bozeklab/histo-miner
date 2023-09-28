#Lucas Sancéré -

################ RMQ

##  maybe create a config outside the submodule but in the configs to make it clearer that the
## submodule shouldn't be touched for inference

## maybe we can do that it is not made to be changed for training as well / This let see


############### SCRIPT

conda deactivate
conda activate mmsegmentation_submodule_test1

cd ./src/models/mmsegmentation/

#May need to export the LIBRARY PATH as follow
# export LD_LIBRARY_PATH="/data/user/miniconda3/envs/hovernet_submodule_test1/lib/:$LD_LIBRARY_PATH"


# Download weigths to add



# Set number of open files limit to 10 000! 
ulimit -n 10000

echo "Run mmsegmentation submodule inference..."


# MUCH MORE COMPLEX as it is using config file from the submodule :(


############## TO MODIFY
#run segmenter inference with 1 GPU
#format is python pythonfile.py configfile.py checkpointsfile.pth outputoptions
python ./tools/test.py \ 
configs/segmenter/segmenter_vit-l_SCC_mask_8x1_640x640_160k_ade20k.py \ 
checkpoints/segmenter_UQ1xTraining_v2023-03-09.pth \ 
--format-only --eval-options "imgfile_prefix=./data/UQ/data/shared/scc/data_conversion_downsampling/downsfactor-40/no_progress/Non-Melanoma/inference_output/"
##############
 


