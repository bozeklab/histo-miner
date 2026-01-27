# Histo-Miner: Tissue Features Extraction With Deep Learning from H&E Images of Squamous Cell Carcinoma Skin Cancer

<div align="center">
[Histo-Miner presentation](#presentation-of-the-pipeline) • [Project Structure](#project-structure) • [Visualization](#visualization) • [Installation](#installation) •  [Usage](#usage) • [Example](#example) •  [Datasets](#datasets) • [Checkpoints](#models-checkpoints) • [Q&A](#models-checkpoints) • [Citation](#citation)

</div>



<br>

This repository contains the code for ["Histo-Miner: Tissue Features Extraction With Deep Learning from H&E Images of Squamous Cell Carcinoma Skin Cancer"](https://www.arxiv.org/abs/2505.04672) paper.


## Presentation of the pipeline

<p align="center">
  <img src="docs/images/pipeline-repr-forgithub.png" width="650">
</p>


Histo-Miner employs convolutional neural networks and vision transformers models for nucleus segmentation and classification as well as tumor region segmentation **(a), (b), (c)**. From these predictions, it generates a compact feature vector summarizing tissue morphology and cellular interactions **(d)**. We used such generated features to classify cSCC patient response to immunotherapy. 


## Project structure 

Here is an explanation of the project structure:

```bash
├── configs                          # All configs file with explanations
│   ├── models                       # Configs for both models inference
│   ├── classification_training      # Configs for classifier training 
│   ├── histo_miner_pipeline         # Configs for the core code of histo-miner
├── docs                             # Images and Videos files
├── example                          # End to end example to run histo-miner
├── scripts                          # Main code for users to run histo-miner 
├── src                              # Functions used for scripts
│   ├── histo-miner                  # All functions from the core code
│   ├── models                       # Submodules of models for inference and training
│   │   ├── hover-net                # hover-net submodule
│   │   ├── mmsegmentation           # segmenter submodule
├── supplements                      # Mathematica notebook for probability of distance overestimaiton calculation
├── visualization                    # Both python and groovy scripts to either reproduce paper figures or to vizualize model inference with qupath   
```

_Note:_ Use the slider to fully read the comments for each section. 

## Visualization

<div align="center">

![](docs/videos/qupath_visualization_v01_crop.gif)

</div>


<p align="center">
  <img src="docs/images/cell_legend.png" width="700">
</p>


<div align="center">
SCC Hovernet and SCC Segmenter nucleus segmentation and classification visualization 
<br> (step <b>(c)</b> from figure above) 
</div>



## Installation

The full pipeline requires 3 environments to work, one for each git submodule and one for the core histo-miner code. We propose hovernet git submodule containing the code of SCC Hovernet model and mmsegmentation git submodule containing the code of SCC Segmenter model. The reason why git submodules are used in this project are detailed in the Q&A section.


### Installation commands

First clone the repository **including its submodules**:
```bash
git clone --recurse-submodules git@github.com:bozeklab/histo-miner.git
```

To use histo-miner you need hardware with CUDA-enable GPUs (NVIDIA GPUs). CPU-only compatible environment are not supported yet. 


```bash
cd histo-miner

# histo-miner env
conda env create -f histo-miner-env.yml
conda activate histo-miner-env
pip install --no-dependencies mrmr-selection==0.2.5

# mmsegmentation submodule env
conda create --name mmsegmentation_submodule python=3.8 -y
conda activate mmsegmentation_submodule
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install click==8.2.1
pip install -U openmim
mim install mmengine
mim install mmcv==2.2.0
mim install mmcv-full==1.7.2
pip install mmsegmentation==1.2.2

# hovernet submodule env
conda env create -f hovernet_submodule.yml
conda activate hovernet_submodule
pip install torch==1.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

conda deactivate
```


### Installation details and alternatives 


You can create environments from scratch instead of using yaml files.


<details>
  <summary>Alternative commands: </summary>

   ```bash
   cd histo-miner

   # histo-miner env
   conda create -n histo-miner-env-nomrmr python=3.10 -y
   conda activate histo-miner-env-nomrmr
   conda install -c conda-forge openslide=3.4.1
   pip install -r ./requirements.txt
   pip install --no-dependencies mrmr-selection==0.2.5

   # mmsegmentation submodule env
   conda create --name mmsegmentation_submodule python=3.8 -y
   conda activate mmsegmentation_submodule
   pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
   pip install click==8.2.1
   pip install -U openmim
   mim install mmengine
   mim install mmcv==2.2.0
   mim install mmcv-full==1.7.2
   pip install mmsegmentation==1.2.2

   # hovernet submodule env
   conda create -n hovernet_submodule python==3.6.12 -y
   conda activate hovernet_submodule
   pip install torch==1.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
   chmod +x src/models/hover_net/setup_condainstall.sh
   yes | ./src/models/hover_net/setup_condainstall.sh
   pip install -r src/models/hover_net/requirements.txt

   conda deactivate
   ```

   The installation can take some time, especially for `conda install -c conda-forge openslide`, `pip install torch` and `yes | ./src/models/hover_net/setup_condainstall.sh` commands.

   If you face problems installing pytorch, check next section. It is also possible to install pytorch no-gpu versions. 

</details>


Further details  are available in the README files of each git submodule.
These files are: 
- For hovernet submodule: `/src/models/hover_net/README.md`
- For mmsegmentation submodule: `/src/models/mmsegmentation/README.md`


## Usage

This section explains how to use the Histo-Miner code. **A complete end-to-end example is also included to help you get started.**


---

### 🔗 Quick Navigation

- [Usage](#Usage)
  - [Models inference: nucleus segmentation and classification](#-models-inference-nucleus-segmentation-and-classification)
  - [Models inference visualization](#-models-inference-visualization)
  - [Tissue Analyser](#-tissue-analyser)
  - [Classification of cSCC response to immunotherapy with pre-defined feature selection](#-classification-of-cscc-response-to-immunotherapy-with-pre-defined-feature-selection)
  - [Classification of cSCC response to immunotherapy with custom feature selection](#-classification-of-cscc-response-to-immunotherapy-with-custom-feature-selection)
- [Example](#example)

**Remark:** You can check how the config files are filled in `example/example-configs/` if you have issues filling configs.  

---


### 🔹 Models inference: nucleus segmentation and classification 

This step performs nucleus segmentation and classification from your input WSIs — corresponding to steps **(a), (b), (c)** in the figure above.

1. Download SCC Segmenter and SCC Hovernet trained weights (see [Datasets](#datasets) for manual download):
   ```bash
   conda activate histo-miner-env
   python src/histo_miner/download_weights.py
   ```
   They are now in `/data/checkpoints/`
2. Configure the files `scc_hovernet.yml` and `scc_segmenter.yml`:
   - Set the input/output paths
   - Set the number of GPUs
   - Set the checkpoints paths (in `./data/checkpoints/`if automatic download)
   - Choose a cache folder 
3. Run the inference (it includes pre-processing of the slides):
   ```bash
   cd scripts
   bash -l main1_hovernet_inference.sh
   bash -l main2_segmenter_inference.sh
   cd ..
   ```
4. Combine the outputs:
   - Copy both outputs in a folder
   - Add the path to this folder to the _inferences_postproc_main_ field in `histo_miner_pipeline.yml` config
5. Run post-processing to correct tumor nuclei classification and reformat files for visualization:
   ```bash
   conda activate histo-miner-env
   python scripts/main3_inferences_postproc.py
   ```

**Output**: One JSON file with segmented and classified nuclei for each input WSI.


---


### 🔹 Models inference visualization 

Visualize the nucleus segmentation and classification as shown in the [Visualization](#visualization) section. 

1. Put the JSON output of "Models inference" step and the corresponding input WSI in the same folder (you can use symbolic links if needed).
2. Ensure both files have the same basename name (excluding extenstion).
3. Open QuPath and open the input WSI inside QuPath. To download QuPath go to: [QuPath website](https://qupath.github.io/).
4. In QuPath:
   - Go to the `Automate` menu → `Script Editor`
   - Load and run the script:
     ```bash
     visualization/qupath_scripts/open_annotations_SCC_Classes.groovy
     ```
5. (Optional) Run the conversion script:
   ```bash
   convert_annotation_to_detection.groovy
   ```
   This helps improve navigation as detection objects are lighter than annotation objects in QuPath.

 

---


### 🔹 Tissue Analyser 

This step computes tissue-relevant features based on previously obtained nucleus segmentations — corresponding to step **(d)** in the figure above.

1. Complete the "Models inference" step.
2. Update the following paths in `histo_miner_pipeline.yml`:
   - `tissue_analyser_main`, folder containing the inference output JSON files
   - `tissue_analyser_output`, path to saving folder 
3. Choose which features to compute using boolean flags in `histo_miner_pipeline.yml`:
   - `calculate_morphologies`, compute or not morphology related features
   - `calculate_vicinity`, compute or not features specifically for cells in tumor vicinity
   - `calculate_distances`, compute or not distance related features (False by default)
4. Run:
   ```bash
   conda activate histo-miner-env
   python scripts/main4_tissue_analyser.py
   ```

**Output**: Structured JSON files with the computed features. 


---


### 🔹 Classification of cSCC response to immunotherapy with pre-defined feature selection   

This step classifies WSIs with tumor regions into responder vs. non-responder for CPI treatment using features selected in the original Histo-Miner paper.

1. Complete the "Models inference" and "Tissue Analyser" steps.
2. Download `Ranking_of_features.json` file ([here](https://uni-koeln.sciebo.de/s/IgiRNgNaAbez9ob) for manual download):
   ```bash
   conda activate histo-miner-env
   python src/histo_miner/download_rank.py
   ```
3. Update the following paths in `histo_miner_pipeline.yml` config:
   - `tissue_analyser_output`, folder containing the tissue analyser output JSONs with correct naming (see 4.)
   - `featarray_folder`, folder to the feature matrix output 
4. Ensure to have "no_response" or "response" caracters in the name of the training json files (depending on the file class). For instance `sample_1_response_analysed.json`.
5. To generate the combined feature matrix and class vectors, run:
   ```bash
   conda activate histo-miner-env
   python scripts/usecase1_collect_features_consistently.py
   ```
6. Update the following parameters in `classification.yml` config:
   - `predefined_feature_selection` must be set to **True**
   - `feature_selection_file`, path to the `Ranking_of_features.json` file (in `./data/feature_rank/`if automatic download)
   - `folders.save_trained_model`, folder to save the model
   - `names.trained_model`, name choosen for the model
   
   Ensure that in `histo_miner_pipeline.yml` config:
   - `nbr_keptfeat` is set to default value: **19**
7. Run:
   ```bash
   python scripts/training/training_classifier.py
   ```
8. Update the following parameters in `classification.yml` config:
   - `inference_input`, path to the folder containing WSI to classify
9. Run:
   ```bash
   python scripts/usecase2_classification_inference.py
   ```

**Output**: Prediction of responder vs non-responder class for each WSI displayed in terminal. 


---


### 🔹 Classification of cSCC response to immunotherapy with custom feature selection

This version performs classification using a new feature selection tailored to your dataset.

1. Complete the "Models inference" and "Tissue Analyser" steps.
2. Update the following paths in `histo_miner_pipeline.yml`:
  - `tissue_analyser_output`, folder containing the tissue analyser output JSONs with correct naming (see next point)
  - `featarray_folder`, folder to the feature matrix output 
3. Ensure to have "no_response" or "response" caracters in the name of the training json files (depending on the file class). For instance `sample_1_response_analysed.json`.
4. To generate the combined feature matrix and class vectors, run:
   ```bash
   conda activate histo-miner-env
   python scripts/usecase1_collect_features_consistently.py
   ```
5. Update `histo_miner_pipeline.yml` config:
   - `classification_evaluation`, path to folder to output cross-validation evaluation
   - `eval_folder`, name of the folder
   Optionally update `classification.yml` for custom parameters.
6. Choose a feature selection method from `scripts/cross_validation/`. We recommand running `featsel_mrmr_std_crossval_samesplits.py`. Run the selected feature method.
7. Update the following parameters in `classification.yml` config:
   - `predefined_feature_selection` must be set to **False**
   - `feature_selection_file`, path to the feature selection numpy file generated in 7. 
   - `folders.save_trained_model`, folder to save the model
   - `names.trained_model`, name choosen for the model

   Importantly update`histo_miner_pipeline.yml` config:
   - `nbr_keptfeat` to the new number of kept features (see infofiles generated in 7. if needed)
8. Run:
   ```bash
   python scripts/training/training_classifier.py
   ```
9. Update the following parameters in `classification.yml` config:
   - `inference_input`, path to the folder containing WSI to classify
10. Run:
   ```bash
   python scripts/usecase2_classification_inference.py
   ```

**Output**: Prediction of responder vs non-responder class for each WSI displayed in terminal. 



---


## Example

An end to end example on how to run the code on one provided WSI is availble in [example](example/) folder. 



## Datasets

* [NucSeg and TumSeg datasets](https://doi.org/10.5281/zenodo.8362592)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8362592.svg)](https://doi.org/10.5281/zenodo.8362592)
* [CPI dataset](https://doi.org/10.5281/zenodo.13986860)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13986860.svg)](https://doi.org/10.5281/zenodo.13986860) 

* [List of features](https://zenodo.org/records/15836085?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjlkZGI1MmI1LWY3OTQtNGMyZS05M2I4LTllY2IzOWNlYzQzMyIsImRhdGEiOnt9LCJyYW5kb20iOiJmNzhkZmZkNmI4MTJhYTExZWIwOWQ5ZTI1OTgyNTJmYSJ9.LYZLjjfWx8RAz_JZnGcNjiUVPU4FWdiUOLqSPxnAULPhw_rRAwBPzRHaMIDH_oAc9P8p4Qx4AwVNEM2QuuECww) 



## Models checkpoints

* [SCC Hovernet and SCC Segmenter models weights](https://doi.org/10.5281/zenodo.13970198)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13970198.svg)](https://doi.org/10.5281/zenodo.13970198)



## Q&A

_**Why this repository is using git submodules and needs 3 conda envs?**_

Git submodules allow for development of SCC Hovernet (hovernet submodule) and SCC Segmenter (mmsegmentation submodule) outside of their initial repositories. This allow to develop them in parallel, not to be influenced by any change done in the original repositories. Also, the code in the submodule contains only the necessary scripts and function to run histo-miner, facilitating readability and reducing repository weight. These modules require python and packages versions that are not compatible with histo-miner core code, so it is not possible to have an environment that fits for the whole repository. <br>

 In short, git submodules allow to treat SCC Hovernet and SCC Segmenter codes as separate projects from histo-miner core code while still allowing for their use within each other. More about git submodule can be found [here](https://git-scm.com/book/en/v2/Git-Tools-Submodules).

_**Why pytorch installation is not included inside the env.yml from hovernet and mmsegmentation submodules?**_

Pytorch versioning depends on the GPUs of your machine. By excluding pytorch installation from the yml files, it allows user to find the pytorch version the most compatible with their software.


## Citation

```
@misc{sancéré2025histominerdeeplearningbased,
      title={Histo-Miner: Deep Learning based Tissue Features Extraction Pipeline from H&E Whole Slide Images of Cutaneous Squamous Cell Carcinoma}, 
      author={Lucas Sancéré and Carina Lorenz and Doris Helbig and Oana-Diana Persa and Sonja Dengler and Alexander Kreuter and Martim Laimer and Anne Fröhlich and Jennifer Landsberg and Johannes Brägelmann and Katarzyna Bozek},
      year={2025},
      eprint={2505.04672},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.04672}, 
}
```
If you use this code or the datasets links please also consider starring the repo to increase its visibility! Thanks 💫


Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
