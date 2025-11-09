# Contributing to CABiNet

First off, thanks for considering contributing to CABiNet! This project aims to provide efficient semantic segmentation for real-time applications, and community contributions are essential to making it better.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to kumaar324@gmail.com.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, command outputs, etc.)
- **Describe the behavior you observed** and what you expected
- **Include your environment details**: OS, Python version, PyTorch version, GPU model
- **Add screenshots or logs** if applicable

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a detailed description** of the suggested enhancement
- **Explain why this enhancement would be useful** to most CABiNet users
- **List any similar features** in other projects if applicable

### Pull Requests

We actively welcome your pull requests:

1. **Fork the repo** and create your branch from `master`
2. **Add tests** if you've added code that should be tested
3. **Ensure the test suite passes** (if we have tests)
4. **Update documentation** for any changed functionality
5. **Follow the existing code style** (use consistent formatting)
6. **Issue the pull request** with a clear description

#### Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update any relevant documentation in the `docs/` folder
3. Reference any related issues in your PR description (e.g., "Fixes #123")
4. The PR will be merged once you have approval from maintainers

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/CABiNet.git
cd CABiNet

# Create conda environment
conda env create -f cabinet_environment.yml --prefix env/cabinet_environment
conda activate env/cabinet_environment

# Install in editable mode
pip3 install -e .
```

## Coding Standards

- **Python Style**: Follow PEP 8 guidelines
- **Docstrings**: Use Google-style docstrings for functions and classes
- **Type Hints**: Add type hints for function parameters and return values
- **Comments**: Write clear comments for complex logic
- **Naming**: Use descriptive variable names (no single letters except in loops)

### Example Code Style

```python
def calculate_miou(predictions: torch.Tensor,
                   targets: torch.Tensor,
                   num_classes: int) -> float:
    """Calculate mean Intersection over Union (mIoU).

    Args:
        predictions: Model predictions of shape (B, C, H, W)
        targets: Ground truth labels of shape (B, H, W)
        num_classes: Number of semantic classes

    Returns:
        Mean IoU score across all classes
    """
    # Implementation here
    pass
```

## Project Structure

- `cabinet/models/` - Model architectures
- `cabinet/datasets/` - Dataset loaders and transforms
- `cabinet/utils/` - Utility functions (loss, optimizer, metrics)
- `scripts/` - Training, evaluation, and demo scripts
- `configs/` - Configuration files for experiments
- `pretrained/` - Pre-trained model weights

## Testing

Currently, the project lacks comprehensive tests. Contributions to add tests are highly welcome!

```bash
# When we have tests, run them like this:
pytest tests/
```

## Documentation

- Update docstrings for any modified functions/classes
- Add comments for complex algorithms
- Update README.md if you change installation or usage
- Consider adding examples in `examples/` folder for new features

## Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and PRs liberally after the first line

### Good Commit Examples

```
Add multi-GPU training support

- Implement DistributedDataParallel wrapper
- Update training script with rank and world_size
- Add documentation for distributed training setup

Fixes #42
```

## Specific Contribution Areas

We particularly welcome contributions in:

### High Priority

- **Pre-trained model weights** for Cityscapes and UAVid
- **Unit tests** for core modules
- **Dockerization** of the project
- **Type hints** throughout the codebase
- **Inference optimization** (TensorRT, ONNX export)

### Medium Priority

- **Additional datasets** (ADE20K, COCO-Stuff, etc.)
- **Training visualization** (TensorBoard/WandB integration)
- **Data augmentation** techniques
- **Export utilities** for mobile/edge deployment
- **Benchmark scripts** comparing with other methods

### Nice to Have

- **Jupyter notebooks** with tutorials
- **Demo applications** (Gradio, Streamlit)
- **Documentation improvements**
- **Performance profiling** tools
- **Architectural variations** of CABiNet

## Getting Help

- **Issues**: Open an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Reach out to kumaar324@gmail.com for specific queries

## Recognition

Contributors will be recognized in:

- README.md Contributors section
- Release notes when applicable
- Academic acknowledgments if contributions are substantial

Thank you for contributing to CABiNet! 🚀
