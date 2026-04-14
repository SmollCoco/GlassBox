import numpy as np

from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series

def train_test_split(
    X: DataFrame, 
    y: Series, 
    test_size: float = 0.2, 
    random_state: int = None, 
    shuffle: bool = True,
    stratify: Series = None
) -> tuple[DataFrame, DataFrame, Series, Series]:
    """Split arrays or matrices into random train and test subsets."""
    n_samples = len(X)
    
    if len(y) != n_samples:
        raise ValueError("X and y must have the same number of samples.")
        
    n_test = int(np.round(n_samples * test_size))
    n_train = n_samples - n_test
    
    indices = np.arange(n_samples)
    
    if stratify is not None:
        strat_arr = stratify.to_numpy()
        classes, y_indices = np.unique(strat_arr, return_inverse=True)
        train_indices_list = []
        test_indices_list = []
        
        rng = np.random.RandomState(random_state)
        for class_index in range(len(classes)):
            class_indices = np.where(y_indices == class_index)[0]
            if shuffle:
                rng.shuffle(class_indices)
            n_class = len(class_indices)
            n_class_test = max(1, int(np.round(n_class * test_size))) if int(np.round(n_class * test_size)) == 0 and n_class >= 2 else int(np.round(n_class * test_size))
            
            test_indices_list.extend(class_indices[:n_class_test])
            train_indices_list.extend(class_indices[n_class_test:])
            
        train_indices = np.array(train_indices_list)
        test_indices = np.array(test_indices_list)
        
        if shuffle:
            rng.shuffle(train_indices)
            rng.shuffle(test_indices)
    else:
        if shuffle:
            rng = np.random.RandomState(random_state)
            rng.shuffle(indices)
            
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
    
    # Reconstruct DataFrames from numpy arrays using rows via indices
    X_arr = np.column_stack([X[col].to_numpy() for col in X.columns])
    
    X_train_arr = X_arr[train_indices]
    X_test_arr = X_arr[test_indices]
    
    # Pack back into DataFrame tracking original columns
    train_data = {col: X_train_arr[:, i].astype(X.dtypes[col]) for i, col in enumerate(X.columns)}
    test_data = {col: X_test_arr[:, i].astype(X.dtypes[col]) for i, col in enumerate(X.columns)}
    
    # Ensure index handling if it exists
    X_train = DataFrame(train_data, columns=X.columns)
    X_test = DataFrame(test_data, columns=X.columns)
    
    y_train = Series(y.to_numpy()[train_indices], name=y.name)
    y_test = Series(y.to_numpy()[test_indices], name=y.name)
    
    return X_train, X_test, y_train, y_test


def train_validation_test_split(
    X: DataFrame, 
    y: Series, 
    train_size: float = 0.7, 
    val_size: float = 0.15, 
    test_size: float = 0.15, 
    random_state: int = None, 
    shuffle: bool = True,
    stratify: Series = None
) -> tuple[DataFrame, DataFrame, DataFrame, Series, Series, Series]:
    """Split arrays or matrices into random train, validation and test subsets."""
    
    if abs((train_size + val_size + test_size) - 1.0) > 1e-5:
        raise ValueError("train_size, val_size, and test_size must sum to 1.0")
        
    n_samples = len(X)
    
    if len(y) != n_samples:
        raise ValueError("X and y must have the same number of samples.")
        
    n_train = int(np.round(n_samples * train_size))
    n_val = int(np.round(n_samples * val_size))
    n_test = n_samples - n_train - n_val
    
    indices = np.arange(n_samples)
    
    if stratify is not None:
        strat_arr = stratify.to_numpy()
        classes, y_indices = np.unique(strat_arr, return_inverse=True)
        train_indices_list = []
        val_indices_list = []
        test_indices_list = []
        
        rng = np.random.RandomState(random_state)
        for class_index in range(len(classes)):
            class_indices = np.where(y_indices == class_index)[0]
            if shuffle:
                rng.shuffle(class_indices)
            n_class = len(class_indices)
            
            n_class_val = int(np.round(n_class * val_size))
            n_class_test = max(1, int(np.round(n_class * test_size))) if int(np.round(n_class * test_size)) == 0 and n_class >= 3 else int(np.round(n_class * test_size))
            n_class_train = n_class - n_class_val - n_class_test
            
            # Make sure no negatives simply bounds check
            if n_class_train < 0:
                n_class_train = 0
            
            train_indices_list.extend(class_indices[:n_class_train])
            val_indices_list.extend(class_indices[n_class_train:n_class_train+n_class_val])
            test_indices_list.extend(class_indices[n_class_train+n_class_val:])
            
        train_indices = np.array(train_indices_list)
        val_indices = np.array(val_indices_list)
        test_indices = np.array(test_indices_list)
        
        if shuffle:
            rng.shuffle(train_indices)
            rng.shuffle(val_indices)
            rng.shuffle(test_indices)
    else:
        if shuffle:
            rng = np.random.RandomState(random_state)
            rng.shuffle(indices)
            
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
    
    X_arr = np.column_stack([X[col].to_numpy() for col in X.columns])
    
    X_train_arr = X_arr[train_indices]
    X_val_arr = X_arr[val_indices]
    X_test_arr = X_arr[test_indices]
    
    train_data = {col: X_train_arr[:, i].astype(X.dtypes[col]) for i, col in enumerate(X.columns)}
    val_data = {col: X_val_arr[:, i].astype(X.dtypes[col]) for i, col in enumerate(X.columns)}
    test_data = {col: X_test_arr[:, i].astype(X.dtypes[col]) for i, col in enumerate(X.columns)}
    
    X_train = DataFrame(train_data, columns=X.columns)
    X_val = DataFrame(val_data, columns=X.columns)
    X_test = DataFrame(test_data, columns=X.columns)
    
    y_train = Series(y.to_numpy()[train_indices], name=y.name)
    y_val = Series(y.to_numpy()[val_indices], name=y.name)
    y_test = Series(y.to_numpy()[test_indices], name=y.name)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
