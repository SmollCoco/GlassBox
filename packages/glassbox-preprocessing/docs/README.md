# GlassBox Preprocessing

Data algorithmic transformation elements strictly complying with the generic Scikit-Learn interface API mappings. Safely bridges `numpandas` natively avoiding array collisions during modeling.

## Key Features
- **Imputers**: `SimpleImputer` computes strategies (mean, median, most_frequent).
- **Scalers**: `StandardScaler`, `MinMaxScaler`, and IQR-based `RobustScaler`.
- **Encoders**: `OneHotEncoder`, `OrdinalEncoder`, `LabelEncoder`.
- **SMOTE**: Oversampling handlers for imbalance datasets.
- **Composition**: `ColumnTransformer` merging discrete execution boundaries natively.

## API Usage

```python
import numpy as np
from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.preprocessing.impute import SimpleImputer
from GlassBox.preprocessing.compose import make_column_transformer
from GlassBox.preprocessing.scale import StandardScaler

df = DataFrame({"age": [20, 25, np.nan], "city": ["A", "B", "A"]})

# Generate preprocessing engine
ct = make_column_transformer(
    (SimpleImputer(strategy='mean', imputation_indicator=True), ["age"]),
    (StandardScaler(), ["age"]),
    remainder='passthrough'
)

transformed_df = ct.fit_transform(df)
```

## Structure
Refer to `docs/UML_Class_Diagram.md` for full attribute typings and mapped methods.
