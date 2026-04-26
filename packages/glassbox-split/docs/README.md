# GlassBox Split

Standalone functional utilities cleanly handling fraction slicing datasets producing reliable testing configurations mapping 2D structures directly onto `numpandas` extraction.

## API Usage

```python
from GlassBox.split.split import train_test_split, train_validation_test_split

# Simple binary split mechanisms 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Complex nested fraction splitting logic
X_tr, X_val, X_te, y_tr, y_val, y_te = train_validation_test_split(X, y, test_size=0.15, val_size=0.15)
```

## Structure
Refer to `docs/UML_Class_Diagram.md` for full attribute typings and mapped methods.
