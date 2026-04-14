from typing import Any

from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series


class Pipeline:
    """A pipeline of transforms with a final estimator.
    
    Sequentially apply a list of transforms and a final estimator.
    It supports integrating resamplers (like SMOTE) dynamically if
    they expose a `fit_resample` method.
    """

    def __init__(self, steps: list[tuple[str, Any]]):
        self.steps = steps

    def fit(self, X: DataFrame, y: Series = None) -> "Pipeline":
        """Fit the model."""
        Xt, yt = X, y
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit_resample"):
                Xt, yt = step.fit_resample(Xt, yt)
            elif hasattr(step, "fit_transform"):
                Xt = step.fit_transform(Xt, yt)
            else:
                Xt = step.fit(Xt, yt).transform(Xt)
                
        final_step = self.steps[-1][1]
        final_step.fit(Xt, yt)
        return self

    def _transform(self, X: DataFrame) -> DataFrame:
        """Internal transform up to the pre-final step."""
        Xt = X
        for name, step in self.steps[:-1]:
            # Resamplers are only applied during training (fit)
            # They just pass data through during predict/transform (if they have transform logic, or we skip them)
            if hasattr(step, "transform"):
                Xt = step.transform(Xt)
        return Xt

    def transform(self, X: DataFrame) -> DataFrame:
        """Transform the data, and apply the final step transform (if available)."""
        Xt = self._transform(X)
        final_step = self.steps[-1][1]
        if hasattr(final_step, "transform"):
            Xt = final_step.transform(Xt)
        return Xt

    def fit_transform(self, X: DataFrame, y: Series = None) -> DataFrame:
        """Fit the pipeline and transform the data."""
        Xt, yt = X, y
        for name, step in self.steps[:-1]:
            if hasattr(step, "fit_resample"):
                Xt, yt = step.fit_resample(Xt, yt)
            elif hasattr(step, "fit_transform"):
                Xt = step.fit_transform(Xt, yt)
            else:
                Xt = step.fit(Xt, yt).transform(Xt)
                
        final_step = self.steps[-1][1]
        if hasattr(final_step, "fit_transform"):
            return final_step.fit_transform(Xt, yt)
        else:
            return final_step.fit(Xt, yt).transform(Xt)

    def predict(self, X: DataFrame) -> Series | list | Any:
        """Apply transforms to the data, and predict with the final estimator."""
        Xt = self._transform(X)
        final_step = self.steps[-1][1]
        return final_step.predict(Xt)
