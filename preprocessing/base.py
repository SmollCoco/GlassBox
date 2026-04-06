from abc import ABC, abstractmethod
from typing import Any

from numpandas.core.dataframe import DataFrame
from numpandas.core.series import Series


class Transformer(ABC):
    """Base class for all preprocessing transformers."""

    @abstractmethod
    def fit(self, X: DataFrame, y: Series | None = None) -> "Transformer":
        """Fit the transformer on the data.

        Parameters
        ----------
        X : DataFrame
            Input data.
        y : Series, optional
            Target values.

        Returns
        -------
        self : Transformer
            Fitted transformer instance.
        """
        pass

    @abstractmethod
    def transform(self, X: DataFrame) -> DataFrame:
        """Transform the data using the fitted parameters.

        Parameters
        ----------
        X : DataFrame
            Data to transform.

        Returns
        -------
        DataFrame
            Transformed data.
        """
        pass

    def fit_transform(self, X: DataFrame, y: Series | None = None) -> DataFrame:
        """Fit the transformer and then transform the data.

        Parameters
        ----------
        X : DataFrame
            Input data.
        y : Series, optional
            Target values.

        Returns
        -------
        DataFrame
            Transformed data.
        """
        return self.fit(X, y).transform(X)
