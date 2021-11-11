# CABiNet

With the increasing demand of autonomous systems, pixelwise semantic segmentation for visual scene understanding needs to be not only accurate but also efficient for potential real-time applications. In this paper, we propose Context Aggregation Network, a dual branch convolutional neural network, with significantly lower computational costs as compared to the state-of-the-art, while maintaining a competitive prediction accuracy. Building upon the existing dual branch architectures for high-speed semantic segmentation, we design a high resolution branch for effective spatial detailing and a context branch with light-weight versions of global aggregation and local distribution blocks, potent to capture both long-range and local contextual dependencies required for accurate semantic segmentation, with low computational overheads. We evaluate our method on two semantic segmentation datasets, namely Cityscapes dataset and UAVid dataset. For Cityscapes test set, our model achieves state-of-the-art results with mIOU of 75.9%, at 76 FPS on an NVIDIA RTX 2080Ti and 8 FPS on a Jetson Xavier NX. With regards to UAVid dataset, our proposed network achieves mIOU score of 63.5% with high execution speed (15 FPS).

![title](imgs/cabinet.jpg)

```
@INPROCEEDINGS{9560977,
  author={Kumaar, Saumya and Lyu, Ye and Nex, Francesco and Yang, Michael Ying},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={CABiNet: Efficient Context Aggregation Network for Low-Latency Semantic Segmentation}, 
  year={2021},
  volume={},
  number={},
  pages={13517-13524},
  doi={10.1109/ICRA48506.2021.9560977}}

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