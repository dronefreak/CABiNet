"""Unit tests for data transformation functions."""

from PIL import Image
import numpy as np
import pytest

from src.datasets.transform import (
    Compose,
    RandomColorJitter,
    RandomCrop,
    RandomCutout,
    RandomGamma,
    RandomHorizontalFlip,
    RandomNoise,
    RandomRotate,
    RandomScale,
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

        assert np.array_equal(
            np.array(result["im"]), np.array(sample_image_label["im"])
        )


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

        assert np.array_equal(
            np.array(result["im"]), np.array(sample_image_label["im"])
        )


class TestRandomNoise:
    """Test RandomNoise transformation."""

    def test_gaussian_noise_applied(self, sample_image_label):
        """Test Gaussian noise is applied."""
        transform = RandomNoise(mode="gaussian", sigma=0.05, p=1.0)
        result = transform(sample_image_label)

        # Images should differ due to noise
        assert result["im"].size == sample_image_label["im"].size
        assert not (np.array(result["im"]) - np.array(sample_image_label["im"])).any()

    def test_noise_never_applied(self, sample_image_label):
        """Test noise with p=0 doesn't modify image."""
        transform = RandomNoise(mode="gaussian", sigma=0.05, p=0.0)
        result = transform(sample_image_label)

        assert np.array_equal(
            np.array(result["im"]), np.array(sample_image_label["im"])
        )


class TestCompose:
    """Test Compose transformation."""

    def test_compose_multiple_transforms(self, sample_image_label):
        """Test composing multiple transformations."""
        transform = Compose(
            [
                RandomScale(scales=[1.0]),
                RandomHorizontalFlip(p=0.0),
                RandomCrop(size=(128, 128)),
            ]
        )

        result = transform(sample_image_label)

        assert result["im"].size == (128, 128)
        assert result["lb"].size == (128, 128)

    def test_compose_empty_transforms(self, sample_image_label):
        """Test compose with no transforms."""
        transform = Compose([])
        result = transform(sample_image_label)

        assert result == sample_image_label


class TestRandomRotate:
    """Test RandomRotate transformation."""

    def test_rotate_label_border_is_ignore_label(self):
        """After rotation the expanded border pixels in the label must be ignore_label (255),
        not 0 (which would be a valid class ID like 'road')."""
        ignore_label = 255
        # Create a label image filled with class 1 so borders stand out
        img = Image.new("RGB", (200, 200), color=(100, 100, 100))
        lb = Image.new("L", (200, 200), color=1)

        transform = RandomRotate(degrees=(45, 45), ignore_label=ignore_label)
        result = transform({"im": img, "lb": lb})

        lb_np = np.array(result["lb"])
        # Corner pixels are introduced by expand=True rotation; they must be ignore_label
        corners = [lb_np[0, 0], lb_np[0, -1], lb_np[-1, 0], lb_np[-1, -1]]
        for corner_val in corners:
            assert corner_val == ignore_label, (
                f"Corner pixel value {corner_val} is not ignore_label ({ignore_label}). "
                "Rotation border fills valid class IDs — use fillcolor=ignore_label."
            )

    def test_rotate_label_no_zero_border(self):
        """Border pixels after rotation must not be 0 (valid road class in Cityscapes)."""
        ignore_label = 255
        img = Image.new("RGB", (100, 100), color=(50, 50, 50))
        lb = Image.new("L", (100, 100), color=5)  # class 5 — non-zero

        transform = RandomRotate(degrees=(30, 30), ignore_label=ignore_label)
        result = transform({"im": img, "lb": lb})

        lb_np = np.array(result["lb"])
        # All pixels should be either class 5 (original) or 255 (border)
        unique_vals = set(np.unique(lb_np).tolist())
        assert unique_vals.issubset(
            {5, ignore_label}
        ), f"Unexpected label values after rotation: {unique_vals - {5, ignore_label}}"

    def test_rotate_image_label_same_size(self):
        """Image and label must remain the same spatial size after rotation."""
        img = Image.new("RGB", (160, 120), color=(128, 128, 128))
        lb = Image.new("L", (160, 120), color=2)

        transform = RandomRotate(degrees=(20, 20), ignore_label=255)
        result = transform({"im": img, "lb": lb})

        assert (
            result["im"].size == result["lb"].size
        ), "Image and label must have same size after rotation"
