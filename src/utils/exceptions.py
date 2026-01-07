"""Custom exceptions for CABiNet."""


class CABiNetError(Exception):
    """Base exception for CABiNet errors."""

    pass


class ModelLoadError(CABiNetError):
    """Raised when model weights fail to load."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to load model from {path}: {reason}")


class DatasetError(CABiNetError):
    """Raised when dataset operations fail."""

    pass


class ConfigurationError(CABiNetError):
    """Raised when configuration is invalid."""

    pass


class TrainingError(CABiNetError):
    """Raised when training fails."""

    pass
