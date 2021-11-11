# CABiNet

With the increasing demand of autonomous systems, pixelwise semantic segmentation for visual scene understanding needs to be not only accurate but also efficient for potential real-time applications. In this paper, we propose Context Aggregation Network, a dual branch convolutional neural network, with significantly lower computational costs as compared to the state-of-the-art, while maintaining a competitive prediction accuracy. Building upon the existing dual branch architectures for high-speed semantic segmentation, we design a high resolution branch for effective spatial detailing and a context branch with light-weight versions of global aggregation and local distribution blocks, potent to capture both long-range and local contextual dependencies required for accurate semantic segmentation, with low computational overheads. We evaluate our method on two semantic segmentation datasets, namely Cityscapes dataset and UAVid dataset. For Cityscapes test set, our model achieves state-of-the-art results with mIOU of __75.9%, at 76 FPS on an NVIDIA RTX 2080Ti and 8 FPS on a Jetson Xavier NX__. With regards to UAVid dataset, our proposed network achieves mIOU score of __63.5% with high execution speed (15 FPS)__. 

An overall model architecture of Context Aggregation Network is shown in the figure below. The spatial and context branches allow for multi-scale feature extraction with significantly low computations. Fusion block (FFM) assists in feature normalization and selection for optimal scene segmentation. The bottleneck in the context branch allows for a deep supervision into the representational learning of the attention blocks.

![title](imgs/cabinet.jpg)

## Results on Cityscapes

Semantic segmentation results on the Cityscapes validation set are shown below. First row consists of the input RGB images; the second row shows the prediction results of SwiftNet (Orsic et al., 2019); the third row shows the predictions from our model and the red boxes show the regions of improvements, and the last row comprises of the ground truth of the input images.

![title](imgs/citys.jpg)


## Results on UAVid

Semantic segmentation results on the UAVid (Lyu et al., 2020) validation set shown below. First column consists of the input RGB images. Second column contains the predictions from our SOTA (Lyu et al., 2020) and the third column contains the predictions from our model. White boxes highlight the regions of improvement over the state-of-the-art (Lyu et al., 2020).

![title](imgs/uavid_r.jpg)


## File Description

* `cab.py` - Contains the implementation of context aggregation block, which is a plug-n-play module, can be added to any PyTorch based network.
* `cabinet.py` - Contains the implementation of the proposed CABiNet model.
* `cityscapes.py` - CityScapes dataloader which requires `cityscapes_info.json` (contains a general description of valid/invalid classes etc.)
* `uavid.py` - UAVid dataloader which requires `UAVid_info.json` (contains a general description of valid/invalid classes etc.)
* `loss.py` - Contains the loss functions for training models.
* `optimizer.py` - Contains the optimizers for training models.
* `mobilenetv3.py` - Contains the implementation of the MobileNetV3 backbones (both Large and Small), the pretrained weights for these backbones can be found under `pretrained/` folder of the repo.
* `transform.py` - Contains data augmentation techniques.
* `train_cabinet_citys.py` - Training code for CABiNet on CityScapes (a similar one can be used for training the model on UAVid, just by changing the imported libraries and path of the datasets).
* `evaluate_cabinet_citys.py` - Evaluation code for trained models, can be used in both multi-scale and single-scale mode.
* `demo.py` - A small demo code for running trained models on custom images.

## Requirements

A conda environment file has been provided in this repo, called `cabinet_environment.yml`. So all you need to setup the repo is to run `conda env create -f cabinet_environment.yml` and everything should be okay. This implementation works with PyTorch>1.0.0 (could work with lower versions, but I have not tested them).

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