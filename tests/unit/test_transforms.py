"""Unit tests for data transformation functions."""

import pytest
import numpy as np
from PIL import Image

from src.datasets.transform import (
    RandomScale,
    RandomHorizontalFlip,
    RandomCrop,
    RandomColorJitter,
    RandomCutout,
    RandomGamma,
    RandomNoise,
    Compose,
)


@pytest.fixture
def sample_image_label():
    """Create sample PIL image and label pair."""
    img = Image.new("RGB", (256, 256), color=(128, 128, 128))
    label = Image.new("L", (256, 256), color=1)
    return {"im": img, "lb": label}


class TestRandomScale:
    """Test RandomScale transformation."""

    def test_random_scale_enlarges(self, sample_image_label):
        """Test RandomScale with scale > 1."""
        transform = RandomScale(scales=[2.0])
        result = transform(sample_image_label)

        assert result["im"].size == (512, 512)
        assert result["lb"].size == (512, 512)

    def test_random_scale_shrinks(self, sample_image_label):
        """Test RandomScale with scale < 1."""
        transform = RandomScale(scales=[0.5])
        result = transform(sample_image_label)

        assert result["im"].size == (128, 128)
        assert result["lb"].size == (128, 128)

    def test_random_scale_multiple_scales(self, sample_image_label):
        """Test RandomScale chooses from multiple scales."""
        transform = RandomScale(scales=[0.5, 1.0, 2.0])
        result = transform(sample_image_label)

        # Size should be one of the valid scales
        valid_sizes = [(128, 128), (256, 256), (512, 512)]
        assert result["im"].size in valid_sizes


class TestRandomHorizontalFlip:
    """Test RandomHorizontalFlip transformation."""

    def test_random_flip_probability_1(self, sample_image_label):
        """Test flip always happens with p=1.0."""
        transform = RandomHorizontalFlip(p=1.0)

        # Create asymmetric image to verify flip
        img = Image.new("RGB", (100, 100))
        img.putpixel((10, 10), (255, 0, 0))
        label = Image.new("L", (100, 100))

        result = transform({"im": img, "lb": label})

        # After flip, the red pixel should be at position (89, 10)
        assert result["im"].getpixel((89, 10)) == (255, 0, 0)

    def test_random_flip_probability_0(self, sample_image_label):
        """Test flip never happens with p=0.0."""
        transform = RandomHorizontalFlip(p=0.0)
        result = transform(sample_image_label)

        assert result["im"].size == sample_image_label["im"].size
        assert result["lb"].size == sample_image_label["lb"].size


class TestRandomCrop:
    """Test RandomCrop transformation."""

    def test_random_crop_exact_size(self, sample_image_label):
        """Test random crop to specific size."""
        transform = RandomCrop(size=(128, 128), pad_if_needed=False)
        result = transform(sample_image_label)

        assert result["im"].size == (128, 128)
        assert result["lb"].size == (128, 128)

    def test_random_crop_with_padding(self):
        """Test random crop with padding when image is too small."""
        img = Image.new("RGB", (64, 64))
        label = Image.new("L", (64, 64))

        transform = RandomCrop(size=(128, 128), pad_if_needed=True, ignore_label=255)
        result = transform({"im": img, "lb": label})

        assert result["im"].size == (128, 128)
        assert result["lb"].size == (128, 128)


class TestRandomColorJitter:
    """Test RandomColorJitter transformation."""

    def test_color_jitter_applied(self, sample_image_label):
        """Test color jitter modifies the image."""
        transform = RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        result = transform(sample_image_label)

        # Image should be modified, label unchanged
        assert result["im"].size == sample_image_label["im"].size
        assert result["lb"] == sample_image_label["lb"]

    def test_color_jitter_none_params(self, sample_image_label):
        """Test color jitter with None parameters does nothing."""
        transform = RandomColorJitter(brightness=None, contrast=None, saturation=None)
        result = transform(sample_image_label)

        assert result["im"] == sample_image_label["im"]


class TestRandomCutout:
    """Test RandomCutout transformation."""

    def test_cutout_applied(self, sample_image_label):
        """Test cutout creates zero region."""
        transform = RandomCutout(p=1.0, size=32)
        result = transform(sample_image_label)

        # Convert to numpy and check for zeros
        img_array = np.array(result["im"])
        assert np.any(img_array == 0), "Cutout should create zero region"

    def test_cutout_never_applied(self, sample_image_label):
        """Test cutout with p=0 doesn't modify image."""
        transform = RandomCutout(p=0.0, size=32)
        result = transform(sample_image_label)

        assert np.array_equal(np.array(result["im"]), np.array(sample_image_label["im"]))


class TestRandomGamma:
    """Test RandomGamma transformation."""

    def test_gamma_applied(self, sample_image_label):
        """Test gamma correction is applied."""
        transform = RandomGamma(gamma_range=(0.8, 1.2), p=1.0)
        result = transform(sample_image_label)

        assert result["im"].size == sample_image_label["im"].size
        assert result["lb"] == sample_image_label["lb"]

    def test_gamma_never_applied(self, sample_image_label):
        """Test gamma with p=0 doesn't modify image."""
        transform = RandomGamma(gamma_range=(0.8, 1.2), p=0.0)
        result = transform(sample_image_label)

        assert np.array_equal(np.array(result["im"]), np.array(sample_image_label["im"]))


class TestRandomNoise:
    """Test RandomNoise transformation."""

    def test_gaussian_noise_applied(self, sample_image_label):
        """Test Gaussian noise is applied."""
        transform = RandomNoise(mode="gaussian", sigma=0.05, p=1.0)
        result = transform(sample_image_label)

        # Images should differ due to noise
        assert result["im"].size == sample_image_label["im"].size
        assert not np.array_equal(np.array(result["im"]), np.array(sample_image_label["im"]))

    def test_noise_never_applied(self, sample_image_label):
        """Test noise with p=0 doesn't modify image."""
        transform = RandomNoise(mode="gaussian", sigma=0.05, p=0.0)
        result = transform(sample_image_label)

        assert np.array_equal(np.array(result["im"]), np.array(sample_image_label["im"]))


class TestCompose:
    """Test Compose transformation."""

    def test_compose_multiple_transforms(self, sample_image_label):
        """Test composing multiple transformations."""
        transform = Compose([
            RandomScale(scales=[1.0]),
            RandomHorizontalFlip(p=0.0),
            RandomCrop(size=(128, 128)),
        ])

        result = transform(sample_image_label)

        assert result["im"].size == (128, 128)
        assert result["lb"].size == (128, 128)

    def test_compose_empty_transforms(self, sample_image_label):
        """Test compose with no transforms."""
        transform = Compose([])
        result = transform(sample_image_label)

        assert result == sample_image_label
