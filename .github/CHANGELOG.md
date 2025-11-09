# Changelog

All notable changes to the CABiNet project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- LICENSE changed from MIT to Apache 2.0
- CODE_OF_CONDUCT.md for community guidelines
- CONTRIBUTING.md for contribution guidelines
- SECURITY.md for security policy and vulnerability reporting
- SUPPORT.md for getting help and support
- CHANGELOG.md for tracking project changes

### Changed

- License updated from MIT License to Apache License 2.0

### Deprecated

- Nothing

### Removed

- Nothing

### Fixed

- Nothing

### Security

- Added security policy and reporting guidelines

## [1.0.0] - 2021-05-XX

### Added

- Initial public release of CABiNet
- Context Aggregation Block (CAB) implementation
- CABiNet model architecture with dual-branch design
- MobileNetV3 backbone implementations (Large and Small)
- Cityscapes dataset dataloader
- UAVid dataset dataloader
- Training script for Cityscapes dataset
- Evaluation script with multi-scale and single-scale support
- Demo script for inference on custom images
- Data augmentation transforms
- Loss functions for training
- Optimizer configurations
- Configuration files for Cityscapes and UAVid
- Conda environment specification
- README with installation and usage instructions

### Performance

- Achieves 75.9% mIOU on Cityscapes test set at 76 FPS (RTX 2080Ti)
- Achieves 8 FPS on Jetson Xavier NX
- Achieves 63.5% mIOU on UAVid with 15 FPS

### Publications

- Accepted to ICRA 2021
- Published in ISPRS Journal of Photogrammetry and Remote Sensing

---

## Version History

### Versioning Scheme

- **Major version (X.0.0)**: Incompatible API changes
- **Minor version (0.X.0)**: New features, backward compatible
- **Patch version (0.0.X)**: Bug fixes, backward compatible

### Upcoming Changes

Future releases may include:

- Pre-trained model weights for Cityscapes and UAVid
- Docker support
- Additional dataset loaders (ADE20K, COCO-Stuff)
- TensorRT and ONNX export utilities
- Training visualization (TensorBoard/WandB)
- Unit tests and CI/CD pipeline
- Type hints throughout codebase
- Jupyter notebook tutorials
- Gradio/Streamlit demo application

### Links

- [Paper (ICRA 2021)](https://doi.org/10.1109/ICRA48506.2021.9560977)
- [Paper (ISPRS Journal)](https://doi.org/10.1016/j.isprsjprs.2021.06.006)
- [GitHub Repository](https://github.com/dronefreak/CABiNet)

---

**Note**: This is a recreated version of the original research codebase. The original implementation was lost due to a system crash. This version may differ slightly from the exact implementation used in the published papers but maintains the core architecture and methodology.

[Unreleased]: https://github.com/dronefreak/CABiNet/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/dronefreak/CABiNet/releases/tag/v1.0.0
