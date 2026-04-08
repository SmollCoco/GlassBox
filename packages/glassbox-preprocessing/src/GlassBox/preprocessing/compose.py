from typing import Any

from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series

from .base import Transformer
from .exceptions import NotFittedError


class ColumnTransformer(Transformer):
    """Applies transformers to columns of an array or numpandas DataFrame.
    
    Parameters
    ----------
    transformers : list of tuples
        List of (name, transformer, columns) tuples specifying the transformer 
        objects to be applied to subsets of the data.
    remainder : {'drop', 'passthrough'}, default='drop'
        By default, only the specified columns in `transformers` are transformed 
        and combined in the output, and the non-specified columns are dropped.
        By specifying `remainder='passthrough'`, all remaining columns that were 
        not specified in `transformers` will be automatically passed through.
    """

    def __init__(self, transformers: list[tuple[str, Transformer, list[str]]], remainder: str = "drop"):
        self.transformers = transformers
        self.remainder = remainder
        self.transformers_: list[tuple[str, Transformer, list[str]]] = []
        self._is_fitted = False
        self.feature_names_in_: list[str] = []

    def fit(self, X: DataFrame, y: Series | None = None) -> "ColumnTransformer":
        self.feature_names_in_ = X.columns
        self.transformers_ = []
        
        for name, trans, cols in self.transformers:
            # Subset the DataFrame for the transformer based on column names
            # Using X[cols] logic:
            subset_X = X[cols] if len(cols) > 0 else DataFrame(columns=[])
            
            # Fit the transformer and store
            if len(cols) > 0:
                trans.fit(subset_X, y)
                
            self.transformers_.append((name, trans, cols))
            
        self._is_fitted = True
        return self

    def transform(self, X: DataFrame) -> DataFrame:
        if not self._is_fitted:
            raise NotFittedError()
            
        all_new_data = {}
        all_new_columns = []
        transformed_cols = set()
        
        for name, trans, cols in self.transformers_:
            if len(cols) == 0:
                continue
            
            subset_X = X[cols]
            transformed_X = trans.transform(subset_X)
            
            for t_col in transformed_X.columns:
                # Add a prefix if the output column shares a name with the input 
                # but might be ambiguous, although standard is to keep it or let transformer dictate.
                # Since transformer dictates, we just adopt it.
                all_new_data[t_col] = transformed_X[t_col].to_numpy()
                all_new_columns.append(t_col)
                
            transformed_cols.update(cols)
            
        # Handle remainder
        if self.remainder == "passthrough":
            for col in X.columns:
                if col not in transformed_cols and col not in all_new_columns:
                    all_new_data[col] = X[col].to_numpy()
                    all_new_columns.append(col)
                    
        return DataFrame(all_new_data, columns=all_new_columns, index=X.index.to_list())


def make_column_transformer(*transformers: tuple[Transformer, list[str]], remainder: str = "drop") -> ColumnTransformer:
    """Construct a ColumnTransformer from the given transformers.
    
    Parameters
    ----------
    *transformers : tuples
        Tuples of the form (transformer, columns)
    remainder : {'drop', 'passthrough'}, default='drop'
        What to do with the remaining columns.
        
    Returns
    -------
    ColumnTransformer
    """
    named_transformers = []
    for i, (trans, cols) in enumerate(transformers):
        name = f"{trans.__class__.__name__.lower()}_{i}"
        named_transformers.append((name, trans, cols))
        
    return ColumnTransformer(named_transformers, remainder=remainder)
