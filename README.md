# Histo-Miner: Tissue Features Extraction With Deep Learning from H&E Images of Squamous Cell Carcinoma Skin Cancer

<div align="center">

[Histo-Miner presentation](#presentation-of-the-pipeline) • [Project Structure](#project-structure) • [Visualization](#visualization) • [Installation](#installation) •  [Usage](#usage) • [Examples](#examples) •  [Datasets](#datasets) • [Checkpoints](#models-checkpoints)  • [Q&A](#models-checkpoints) • [Citation](#citation)  

</div>



<br>
This repository contains the code for ["Histo-Miner: Tissue Features Extraction With Deep Learning from H&E Images of Squamous Cell Carcinoma Skin Cancer"](https://www.arxiv.org/abs/2505.04672) paper.


## Presentation of the pipeline

<p align="center">
  <img src="docs/images/pipeline-repr-forgithub.png" width="650">
</p>


Histo-Miner employs convolutional neural networks and vision transformers models for nucleus segmentation and classification as well as tumor region segmentation **(a), (b), (c)**. From these predictions, it generates a compact feature vector summarizing tissue morphology and cellular interactions. **(d)** We used such generated features to classify cSCC patient response to immunotherapy. 


## Project structure 

Here is an explanation of the project structure:

```bash
├── configs                          # All configs file with explanations
│   ├── models                       # example configs for both models inference
│   ├── classification_training      # Configs for classifier training 
│   ├── histo_miner_pipeline         # Configs for the core code of histo-minerent
├── docs                             # Documentation files (in addition to this main README.md)
├── scripts                          # Main code for users to run Histo-Miner 
├── src                              # Functions used for scripts
│   ├── histo-miner                  # All functions from the core code (everything except deep learning)
│   ├── models                       # Submodules of models for inference and training
│   │   ├── hover-net                # hover-net submodule, simplification of original code to fit histo-miner needs
│   │   ├── mmsegmentation           # segmenter submodule, simplification of original code to fit histo-miner needs
├── supplements                      # Mathematica notebook for probability of distance overestimaiton calculation
├── visualization                    # Both python and groovy scripts to either reproduce paper figures or to vizualize model inference with qupath   s
```

_Note:_ Use the slider to fully read the comments for each section. 

## Visualization

<div align="center">

![](docs/videos/qupath_visualization_v01_crop.gif)

</div>

<div align="center">
SCC Hovernet and SCC Segmenter nucleus segmentation and classification visualization 
(step <b>(c)</b> from figure above) 
</div>



## Installation

The full pipeline requires 3 environments to work, one for each git submodule and one for the core histo-miner code. We propose hovernet git submodule containing the code of SCC Hovernet model and mmsegmentation git submodule containing the code of SCC Segmenter model. The reason why git submodules are used in this project are detailed in the Q&A section.


### Installation commands

The easiest way to install all these environments is from the yml files of the submodules conda envs and the requirement file of the core histo-miner code:

```bash
# histo-miner env
conda create -n histo-miner-env python=3.10
conda activate histo-miner-env
conda install -c conda-forge openslide=3.4.1
pip install -r ./requirements.txt
pip install --no-dependencies mrmr-selection==0.2.5
conda deactivate

# hovernet submodule env
conda env create -f src/models/hover_net/hovernet_submodule.yml
conda activate hovernet_submodule
pip install torch==1.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
conda deactivate

# mmsegmentation submodule env
conda env create -f src/models/mmsegmentation/mmsegmentation_submodule.yml
conda activate mmsegmentation_submodule
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
conda deactivate
```

If you face problems installing pytorch, check next section. It is also possible to install pytorch no-gpu versions. The installation can take some time, especially for `conda install -c conda-forge openslide` and `pip install torch` commands.


### Installation details and alternatives 

Further details as well as alternative installation methods are available in the README files of each git submodule.
These files are: 
- For hovernet submodule: `/src/models/hover_net/README.md`
- For mmsegmentation submodule: `/src/models/mmsegmentation/README.md`


## Usage

`Final version on 15/05/25`

Here we present how to use histo-miner code. **A complete end-to-end example is also included in next paragraph to facilitate usage.**

- [Usage](#Usage)
  - [Models inference: nucleus segmentation and classification](#models-inference-nucleus-segmentation-and-classification)
  - [Models inference visualization](#models-inference-visualization)
  - [Tissue Analyser](#tissue-analyser)
  - [Classification of cSCC response to immunotherapy with pre-defined feature selection](#classification-of-cscc-response-to-immunotherapy-with-pre-defined-feature-selection)
  - [Classification of cSCC response to immunotherapy with custom feature selection](#classification-of-cscc-response-to-immunotherapy-with-custom-feature-selection)
- [Examples](#examples)


### Models inference: nucleus segmentation and classification 

Here we will described how to obtained nucleus segmentation and classification from your input WSI. It corresponds to steps **(a), (b), (c)** from the figure above.

- Download the SCC Segmenter and SCC Hovernet trained weights (see [Datasets](#datasets))
- Fill the models configs (`scc_hovernet.yml` and `scc_segmenter.yml`) to indicate the paths to the different files needed and the number of gpus used for inference,
- Run: `sh scripts/main1_hovernet_inference.sh`,
- Run: `sh scripts/main2_segmenter_inference.sh`,
- Put both inference outputs on the same folder, add the path of this folder to `histo_miner_pipeline.yml` config file on the _inferences_postproc_main_ setting,
- Run: `python scripts/main3_inferences_postproc.py`.

The json files finally obtained contain the nucleus classified and segmented for all WSIs of the input folder. 


### Models inference visualization 

Here we will explain how to visualize the nucleus segmentation and classification as shown in [Visualization](#visualization). 

- Put the json file of nucleus segmentation and classification obtained previsouly and the corresponding input WSI in the same folder, and rename if needed that they both have the same name (only extension change). You can use symbolic links to avoid copying,
- Open QuPath and open the input WSI inside QuPath. To download QuPath go to: [QuPath website](https://qupath.github.io/),
- Open the script editor (Automate menu on top), select the `/visualization/qupath_scripts/open_annotations_SCC_Classes.groovy` file and run it.

You can use the 2 conversion scripts to make navigation easy. In fact, detections object are lighter than annotation in QuPath and `convert_annotation_to_detection.groovy` will allow for easier navigation. 


### Tissue Analyser 

Here we will described how to calculate tissue relevant features based on the previously obtained nucleus segmentation and classification. It corresponds to step **(d)** from the figure above.

- First follow the steps from "Models inference: nucleus segmentation and classification",
- Add the paths to the folder containing the segmentation jsons (_tissue_analyser_main_ setting) and the path the output folder (_tissue_analyser_output_ setting) in the  `histo_miner_pipeline.yml` config file,
- Decide wich features to compute based on the choice of _calculate_morphologies_ , _calculate_vicinity_ and _calculate_distances_ boolean parameters in  `histo_miner_pipeline.yml` config file,
- Run: 'python scripts/main4_tissue_analyser.py'.

The structured json files obtained contain the features values computed.


### Classification of cSCC response to immunotherapy with pre-defined feature selection   

Here we perform binary classification of WSI with tumor region into responder and non-responder to a futur immunotherapy (CPI) treatment. We will use the same selected feature as in the Histo-Miner paper. 


- First follow the steps from "Models inference: nucleus segmentation and classification" and "Tissue Analyser",
- Download `Ranking_of_features.json` file from CPI dataset (see [Datasets](#datasets)). We will use these pre-defined selected features to do our classification later on,
- Add the paths to the folder containing the features jsons (_tissue_analyser_output_ setting) and the path to the post-processed features output folder (_featarray_folder_ setting) in the  `histo_miner_pipeline.yml` config file,
- Run: `python scripts/usecase1_collect_features_consistently.py` to create one matrix with all samples feature,
- _Writting of next steps in progress_ 


### Classification of cSCC response to immunotherapy with custom feature selection

Here we perform binary classification of WSI with tumor region into responder and non-responder to a futur immunotherapy (CPI) treatment. We will perform a new feature selection to fit more with our dataset.

- First follow the steps from "Models inference: nucleus segmentation and classification" and "Tissue Analyser",
- Add the paths to the folder containing the features jsons (_tissue_analyser_output_ setting) and the path to the post-processed features output folder (_featarray_folder_ setting) in the  `histo_miner_pipeline.yml` config file,
- Run: `python scripts/usecase1_collect_features_consistently.py` to create one matrix with all samples feature,
- Choose which feature selection method you want to compute based on the scripts in `scripts/cross_validation/`. We recommand running `featsel_mrmr_std_crossval_samesplits.py`,
- Add the path to the folder to output cross-validation evaluation (_classification_evaluation_ setting) and choose its name (_eval_folder_ setting) in `histo_miner_pipeline.yml` config file. Optionnaly you can also modify `classification.yml` config file to add/modify any custom parameters,
- Run 'python scripts/cross_validation/name_of_choosen_method.py',
- In the folder `infofiles` newly created as output, you will find a .txt file with selected feature names.
- _Writting of next steps in progress_ 



## Examples 

-> Under construction: `Available on 22/05/25 or before` 


## Datasets

* [NucSeg and TumSeg datasets](https://doi.org/10.5281/zenodo.8362593)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8362593.svg)](https://doi.org/10.5281/zenodo.8362593)
* [CPI dataset](https://doi.org/10.5281/zenodo.13986860)  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13986860.svg)](https://doi.org/10.5281/zenodo.13986860) 



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
