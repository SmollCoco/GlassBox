# Preprocessing Package

The `preprocessing` package offers an API entirely analogous to Scikit-Learn's estimators, featuring `fit`, `transform`, and `fit_transform` methods fully tailored to `numpandas` architecture.

## Transformers Provided
- **Imputers**: `SimpleImputer` computes strategies (mean, median, most_frequent, constant) and optionally produces missingness indicators appending column aliases like `col_imputed`.
- **Scalers**: `StandardScaler` (Unit variance scaling), `MinMaxScaler` ([0, 1] scaling), and `RobustScaler` (IQR-based).
- **Encoders**:
  - `OneHotEncoder`: Converts categories into sparse indicator columns. Unknown categories encountered in production can be gracefully managed using `handle_unknown='ignore'`.
  - `OrdinalEncoder`: Assigns numerical integers to qualitative classes on 2D Input Variables `X`.
  - `LabelEncoder`: Specially designed for 1D target `y` Series.
- **Composer**: `ColumnTransformer` allows building an end-to-end multi-step pipeline combining previous processing onto a select list of characteristics in the dataset.

## Usage

```python
from GlassBox.numpandas import DataFrame
from GlassBox.preprocessing import SimpleImputer, StandardScaler, make_column_transformer

df = DataFrame({"age": [20, 25, np.nan], "city": ["A", "B", "A"]})

# Define Preprocessing
ct = make_column_transformer(
    (SimpleImputer(strategy='mean', imputation_indicator=True), ["age"]),
    (StandardScaler(), ["age"]),
    remainder='passthrough'
)

# Learn parameters and alter dataset simultaneously
transformed_df = ct.fit_transform(df)
```

## Structure
- `preprocessing/base.py`: The root interface defining estimators.
- `preprocessing/impute.py`, `preprocessing/scale.py`, `preprocessing/encode.py`: Distinct families of algorithms.
- `preprocessing/compose.py`: ColumnTransformer logic.
- `preprocessing/UML_Class_Diagram.md`: Class architecture.
