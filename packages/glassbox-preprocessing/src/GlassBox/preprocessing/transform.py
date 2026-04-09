from typing import Callable, Any

from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series

from .base import Transformer


class FunctionTransformer(Transformer):
    """Constructs a transformer from an arbitrary callable.
    
    A FunctionTransformer forwards its X (and optionally y) arguments to a
    user-defined function or function object and returns the result of this
    function.
    
    Parameters
    ----------
    func : callable, optional
        The callable to use for the transformation.
    kw_args : dict, optional
        Dictionary of additional keyword arguments to pass to func.
    """
    
    def __init__(self, func: Callable[[DataFrame, Any], DataFrame] | None = None, **kw_args):
        self.func = func
        self.kw_args = kw_args

    def fit(self, X: DataFrame, y: Series | None = None) -> "FunctionTransformer":
        """Fit transformer by checking X.
        
        If func is None, this simply returns self.
        """
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        """Transform X using the forward function.
        
        Parameters
        ----------
        X : DataFrame
            Input data.
            
        Returns
        -------
        DataFrame
            Transformed data.
        """
        if self.func is None:
            return X
        return self.func(X, **self.kw_args)
