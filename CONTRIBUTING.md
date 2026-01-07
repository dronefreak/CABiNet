# Contributing to CABiNet

Thank you for your interest in contributing to CABiNet! This document provides guidelines and best practices for contributing to the project.

## Development Setup

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/dronefreak/CABiNet.git
cd CABiNet
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

4. Install development tools:
```bash
pip install pytest black isort flake8 mypy
```

## Code Quality

### Style Guidelines

We follow PEP 8 with some modifications:
- Maximum line length: 100 characters
- Use Black for code formatting
- Use isort for import sorting
- Type hints are encouraged but not required

### Pre-commit Checks

Before submitting a PR, run these checks:

```bash
# Format code
black src/
isort src/

# Lint code
flake8 src/

# Type check
mypy src/

# Run tests
pytest tests/
```

### Code Organization

- **Models**: `src/models/` - Neural network architectures
  - Shared layers in `src/models/layers/`
  - Constants in `src/models/constants.py`
- **Datasets**: `src/datasets/` - Data loading and augmentation
- **Utils**: `src/utils/` - Helper functions and utilities
- **Scripts**: `src/scripts/` - Training, evaluation, visualization scripts

### Naming Conventions

- **Classes**: PascalCase (e.g., `CABiNet`, `HardSwish`)
- **Functions/Methods**: snake_case (e.g., `forward`, `get_params`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `OHEM_DIVISOR`, `DEFAULT_SCORE_THRESHOLD`)
- **Private methods**: Prefix with underscore (e.g., `_initialize_weights`)

## Pull Request Process

1. **Fork and Branch**
   - Fork the repository
   - Create a feature branch: `git checkout -b feature/your-feature-name`

2. **Make Changes**
   - Write clear, concise commit messages
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   - Ensure all tests pass
   - Add new tests for new features
   - Check code style compliance

4. **Submit PR**
   - Push to your fork
   - Open a Pull Request against `main` branch
   - Fill out the PR template completely
   - Link any related issues

5. **Code Review**
   - Address reviewer feedback
   - Keep PR focused and reasonably sized
   - Be responsive to questions

## Commit Message Format

Follow the Conventional Commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code refactoring
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**
```
feat(models): Add support for MobileNetV3-Large backbone

- Implement MobileNetV3-Large architecture
- Add pretrained weight loading
- Update model configuration

Closes #123
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run tests matching pattern
pytest -k "test_forward"
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Use descriptive test function names
- Include docstrings explaining what is being tested

Example:
```python
def test_cabinet_forward_shape():
    """Test CABiNet forward pass output shape."""
    model = CABiNet(n_classes=19, mode="large")
    x = torch.randn(2, 3, 512, 512)
    out, out16 = model(x)

    assert out.shape == (2, 19, 512, 512)
    assert out16.shape == (2, 19, 512, 512)
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: int, param2: str) -> bool:
    """Short description of function.

    Longer description if needed, explaining the function's purpose,
    behavior, and any important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Raises:
        ValueError: When param1 is negative

    Example:
        >>> result = function_name(5, "test")
        >>> print(result)
        True
    """
    pass
```

### Adding New Features

When adding new features:
1. Add comprehensive docstrings
2. Update README.md if user-facing
3. Add configuration examples
4. Include usage examples
5. Write tests

## Performance Considerations

- Profile code before optimizing
- Use `torch.amp` for mixed precision training
- Leverage `DataLoader` prefetching
- Cache expensive computations when possible
- Use appropriate batch sizes

## Questions?

- Open an issue for bugs or feature requests
- Use discussions for questions and ideas
- Check existing issues before creating new ones

Thank you for contributing to CABiNet! 🚀
