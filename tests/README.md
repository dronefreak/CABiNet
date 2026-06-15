# CABiNet Test Suite

This directory contains the test suite for the CABiNet semantic segmentation project.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures and configuration
├── unit/                    # Unit tests for individual components
│   ├── test_models.py       # Model architecture tests
│   ├── test_loss.py         # Loss function tests
│   └── test_transforms.py   # Data augmentation tests
├── integration/             # Integration tests
│   └── test_training_pipeline.py  # End-to-end training tests
└── fixtures/                # Test data and fixtures
```

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Category
```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# Specific test file
pytest tests/unit/test_models.py

# Specific test function
pytest tests/unit/test_models.py::TestCABiNetModel::test_cabinet_forward_shape
```

### Run with Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

This generates an HTML coverage report in `htmlcov/index.html`.

### Run in Parallel
```bash
pytest tests/ -n auto
```

Uses all available CPU cores for faster testing.

### Run with Verbose Output
```bash
pytest tests/ -v
```

## Test Markers

Tests can be marked for selective execution:

```python
@pytest.mark.slow  # Slow tests
@pytest.mark.gpu   # Requires GPU
@pytest.mark.integration  # Integration tests
```

Run specific markers:
```bash
pytest tests/ -m "not slow"  # Skip slow tests
pytest tests/ -m gpu        # Run only GPU tests
```

## Writing Tests

### Unit Test Example

```python
def test_model_forward():
    """Test model forward pass."""
    model = CABiNet(n_classes=19, mode="large")
    x = torch.randn(1, 3, 512, 512)

    out, out16 = model(x)

    assert out.shape == (1, 19, 512, 512)
    assert not torch.isnan(out).any()
```

### Using Fixtures

```python
def test_with_fixture(sample_image, num_classes):
    """Test using shared fixtures."""
    model = CABiNet(n_classes=num_classes, mode="large")
    out, _ = model(sample_image)

    assert out.shape[0] == sample_image.shape[0]
```

### Parametrized Tests

```python
@pytest.mark.parametrize("mode", ["large", "small"])
def test_multiple_modes(mode, num_classes):
    """Test with different model modes."""
    model = CABiNet(n_classes=num_classes, mode=mode)
    # ... test code
```

## Continuous Integration

Tests are automatically run on:
- Every push to main/develop branches
- Every pull request
- Multiple Python versions (3.8, 3.9, 3.10, 3.11)
- Multiple OS (Ubuntu, macOS)

See `.github/workflows/ci.yml` for CI configuration.

## Test Coverage Goals

- **Unit Tests**: >80% coverage
- **Integration Tests**: Critical paths covered
- **Model Tests**: All forward/backward passes tested
- **Loss Tests**: All loss functions validated

## Performance Tests

Use the performance profiler for benchmarking:

```python
from src.utils.profiler import PerformanceProfiler

profiler = PerformanceProfiler(model)
results = profiler.run_full_benchmark()
```

## Contributing

When adding new features:
1. Write tests first (TDD)
2. Ensure tests pass locally
3. Check coverage: `pytest --cov=src`
4. Run pre-commit hooks: `pre-commit run --all-files`

See `CONTRIBUTING.md` for more details.
