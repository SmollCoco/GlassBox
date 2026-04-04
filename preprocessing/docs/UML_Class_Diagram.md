# Preprocessing Package UML Class Diagram

```mermaid
classDiagram
    class Transformer {
        <<Abstract>>
        +fit(X, y) Transformer
        +transform(X) DataFrame
        +fit_transform(X, y) DataFrame
    }
    
    class SimpleImputer {
        +strategy: str
        +imputation_indicator: bool
        +fit(X, y)
        +transform(X)
    }
    
    class StandardScaler {
        +fit(X, y)
        +transform(X)
    }
    class MinMaxScaler {
        +fit(X, y)
        +transform(X)
    }
    class RobustScaler {
        +fit(X, y)
        +transform(X)
    }
    
    class OneHotEncoder {
        +handle_unknown: str
        +fit(X, y)
        +transform(X)
    }
    class OrdinalEncoder {
        +handle_unknown: str
        +fit(X, y)
        +transform(X)
    }
    class LabelEncoder {
        +fit(y)
        +transform(y)
    }
    
    class FunctionTransformer {
        +func: callable
        +fit(X, y)
        +transform(X)
    }
    
    class ColumnTransformer {
        +transformers: list
        +remainder: str
        +fit(X, y)
        +transform(X)
    }
    
    Transformer <|-- SimpleImputer
    Transformer <|-- StandardScaler
    Transformer <|-- MinMaxScaler
    Transformer <|-- RobustScaler
    Transformer <|-- OneHotEncoder
    Transformer <|-- OrdinalEncoder
    Transformer <|-- FunctionTransformer
    Transformer <|-- ColumnTransformer
```

## Description
- **Transformer**: The abstract base class that establishes the unified `fit()`, `transform()`, and `fit_transform()` interface.
- **ColumnTransformer**: Combines multiple transformers applied to different specific columns of a `DataFrame`, similar to Scikit-Learn.
- **LabelEncoder vs OrdinalEncoder**: Note that `LabelEncoder` expects a 1D target `y` vector (or `Series`), while `OrdinalEncoder` processes categorical attributes defined in 2D inputs `X`.
