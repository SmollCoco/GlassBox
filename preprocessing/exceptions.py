class PreprocessingError(Exception):
    """Base exception for preprocessing errors."""
    pass


class NotFittedError(PreprocessingError):
    """Exception raised when a transformer is used before being fitted."""
    def __init__(self, message="This Transformer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."):
        super().__init__(message)


class DimensionalityError(PreprocessingError):
    """Exception raised when input shape is mismatched or unexpected."""
    pass
