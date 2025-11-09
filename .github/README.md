# CABiNet

With the increasing demand of autonomous systems, pixelwise semantic segmentation for visual scene understanding needs to be not only accurate but also efficient for potential real-time applications. In this paper, we propose Context Aggregation Network, a dual branch convolutional neural network, with significantly lower computational costs as compared to the state-of-the-art, while maintaining a competitive prediction accuracy. Building upon the existing dual branch architectures for high-speed semantic segmentation, we design a high resolution branch for effective spatial detailing and a context branch with light-weight versions of global aggregation and local distribution blocks, potent to capture both long-range and local contextual dependencies required for accurate semantic segmentation, with low computational overheads. We evaluate our method on two semantic segmentation datasets, namely Cityscapes dataset and UAVid dataset. For Cityscapes test set, our model achieves state-of-the-art results with mIOU of **75.9%, at 76 FPS on an NVIDIA RTX 2080Ti and 8 FPS on a Jetson Xavier NX**. With regards to UAVid dataset, our proposed network achieves mIOU score of **63.5% with high execution speed (15 FPS)**.

An overall model architecture of Context Aggregation Network is shown in the figure below. The spatial and context branches allow for multi-scale feature extraction with significantly low computations. Fusion block (FFM) assists in feature normalization and selection for optimal scene segmentation. The bottleneck in the context branch allows for a deep supervision into the representational learning of the attention blocks.

![title](imgs/cabinet.jpg)

## Results on Cityscapes

Semantic segmentation results on the Cityscapes validation set are shown below. First row consists of the input RGB images; the second row shows the prediction results of SwiftNet (Orsic et al., 2019); the third row shows the predictions from our model and the red boxes show the regions of improvements, and the last row comprises of the ground truth of the input images.

![title](imgs/citys.jpg)

## Results on UAVid

Semantic segmentation results on the UAVid (Lyu et al., 2020) validation set shown below. First column consists of the input RGB images. Second column contains the predictions from our SOTA (Lyu et al., 2020) and the third column contains the predictions from our model. White boxes highlight the regions of improvement over the state-of-the-art (Lyu et al., 2020).

![title](imgs/uavid_r.jpg)

## Setup Requirements

A conda environment file has been provided in this repo, called `cabinet_environment.yml`. So all you need to setup the repo is to run `conda env create -f cabinet_environment.yml` and everything should be okay. This implementation works with PyTorch>1.0.0 (could work with lower versions, but I have not tested them). A more systematic method of setting up the project is given below:

```
mkdir env/
conda env create -f cabinet_environment.yml --prefix env/cabinet_environment
conda activate env/cabinet_environment
pip3 install -e .
```

In the above process we basically use the prefix for this conda environment so that it is local to this repo. Also, since this project is packaged, its advisabele to install it in editable mode via `pip3 install -e .` inside the conda environment.

## File Description

A quick overview of the different files and packages inside this project.

### Core

Contains model implementations under `models/`, dataloaders under `datasets/` and general project `utils/`.

- `models/cab.py` - Contains the implementation of context aggregation block, which is a plug-n-play module, can be added to any PyTorch based network.
- `models/cabinet.py` - Contains the implementation of the proposed CABiNet model.
- `models/mobilenetv3.py` - Contains the implementation of the MobileNetV3 backbones (both Large and Small), the pretrained weights for these backbones can be found under `pretrained/` folder of the repo.
- `datasets/cityscapes.py` - CityScapes dataloader which requires `cityscapes_info.json` (contains a general description of valid/invalid classes etc.)
- `datasets/uavid.py` - UAVid dataloader which requires `UAVid_info.json` (contains a general description of valid/invalid classes etc.)
- `datasets/transform.py` - Contains data augmentation techniques.
- `utils/loss.py` - Contains the loss functions for training models.
- `utils/optimizer.py` - Contains the optimizers for training models.

### Scripts

Contains training, validation and demo scripts model analysis.

- `scripts/train.py` - Training code for CABiNet on CityScapes (a similar one can be used for training the model on UAVid, just by changing the imported libraries and path of the datasets).
- `scripts/evaluate.py` - Evaluation code for trained models, can be used in both multi-scale and single-scale mode.
- `scripts/demo.py` - A small demo code for running trained models on custom images.

### Configs

- `configs/train_citys.json` - Training and validation config file for CABiNet model on CityScapes dataset. Please use this file to manage input/output directories, dataset paths and other training parameters.
- `configs/train_uavid.json` - Training and validation config file for CABiNet model on UAVid dataset. Please use this file to manage input/output directories, dataset paths and other training parameters.
- `configs/cityscapes_info.json` - Valid/invalid label information about CityScapes dataset.
- `configs/UAVid_info.json` - Valid/invalid label information about UAVid dataset.

## Training/Evaluation on CityScapes and UAVid

Well the pipeline should be pretty easy, you need to download the CityScapes dataset from [here](https://www.cityscapes-dataset.com/downloads/). Look for `gtFine_trainvaltest.zip (241MB)` for the GT and
`leftImg8bit_trainvaltest.zip (11GB)` for the input corresponding RGB images. For UAVid you can download the dataset from [here](https://uavid.nl/), under `Downloads`. Once the datasets are downloaded, extract the .zip files and specify the dataset paths in the appropriate config files under `configs/`

Then simply run the following commands:

```
export CUDA_VISIBLE_DEVICES=0, # or 1, 2 or 3 (depending upon which device you want to use in case there are multiple.)
python3 scripts/train.py --config configs/train_citys.json
```

The train script executes and trains the model, and saves the model weights 10 times during the training, depending upon the best mIOU score obtained on the validation set during training.

**Pre-trained CABiNet models coming soon !!!**

## Issues and Pull Requests

Please feel free to create PRs and/or send me issues directly to `kumaar324@gmail.com`. I will be happy to help, but I might not be available a lot of times, still I will try my best.

# Citation

If you find this work helpful, please consider citing the following articles:

```
@INPROCEEDINGS{9560977,
  author={Kumaar, Saumya and Lyu, Ye and Nex, Francesco and Yang, Michael Ying},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  title={CABiNet: Efficient Context Aggregation Network for Low-Latency Semantic Segmentation},
  year={2021},
  pages={13517-13524},
  doi={10.1109/ICRA48506.2021.9560977}}

```

and

```
@article{YANG2021124,
title = {Real-time Semantic Segmentation with Context Aggregation Network},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {178},
pages = {124-134},
year = {2021},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2021.06.006},
url = {https://www.sciencedirect.com/science/article/pii/S0924271621001647},
author = {Michael Ying Yang and Saumya Kumaar and Ye Lyu and Francesco Nex},
keywords = {Semantic segmentation, Real-time, Convolutional neural network, Context aggregation network}
}
```
