import numpy as np
from GlassBox.numpandas.core.dataframe import DataFrame
from GlassBox.numpandas.core.series import Series

class SMOTE:
    """Synthetic Minority Over-sampling Technique (SMOTE).
    
    Supports both numerical and categorical features implicitly by applying 
    SMOTENC-like logic for mixed type datasets.
    """
    
    def __init__(self, k_neighbors: int = 5, random_state: int = None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
    def fit_resample(self, X: DataFrame, y: Series) -> tuple[DataFrame, Series]:
        """Resamples the dataset making class distribution balanced."""
        rng = np.random.RandomState(self.random_state)
        
        y_arr = y.to_numpy()
        classes, counts = np.unique(y_arr, return_counts=True)
        
        if len(classes) <= 1:
            return X, y  # Nothing to balance
            
        majority_count = np.max(counts)
        minority_class = classes[np.argmin(counts)]
        
        # Determine numerical vs categorical columns
        num_cols = []
        cat_cols = []
        for i, col in enumerate(X.columns):
            if np.issubdtype(X.dtypes[col], np.number):
                num_cols.append((i, col))
            else:
                cat_cols.append((i, col))
                
        num_indices = [i for i, _ in num_cols]
        cat_indices = [i for i, _ in cat_cols]
        
        X_arr = np.column_stack([X[col].to_numpy() for col in X.columns])
        
        # Calculate standard deviation for numerical penalty in categorical distance
        penalty = 0.0
        if num_indices and cat_indices:
            num_data = X_arr[:, num_indices].astype(float)
            stds = np.std(num_data, axis=0)
            penalty = np.median(stds) if len(stds) > 0 else 1.0
            if penalty == 0:
                penalty = 1.0
                
        # Get minority instances
        minority_mask = (y_arr == minority_class)
        minority_X = X_arr[minority_mask]
        n_minority = len(minority_X)
        
        n_synthetic_needed = majority_count - n_minority
        if n_synthetic_needed <= 0:
            return X, y
            
        synthetic_X = []
        synthetic_y = []
        
        # Calculate pairwise distance for minority vs minority
        # We only need neighbors among minority
        
        for i in range(n_minority):
            current_x = minority_X[i]
            
            # Compute distance
            distances = np.zeros(n_minority)
            for j in range(n_minority):
                if i == j:
                    distances[j] = np.inf
                    continue
                
                other_x = minority_X[j]
                
                dist = 0.0
                if num_indices:
                    diff = current_x[num_indices].astype(float) - other_x[num_indices].astype(float)
                    dist += np.sum(diff ** 2)
                    
                if cat_indices:
                    # Hamming distance scaled by penalty squared
                    diff_cat = (current_x[cat_indices] != other_x[cat_indices])
                    dist += np.sum(diff_cat) * (penalty ** 2)
                    
                distances[j] = np.sqrt(dist)
                
            # Get k nearest neighbors
            # If n_minority - 1 < k_neighbors, take all available
            k = min(self.k_neighbors, n_minority - 1)
            if k <= 0:
                # Cannot generate using neighbors if only 1 minority instance
                # Just duplicate it
                for _ in range(n_synthetic_needed):
                    synthetic_X.append(current_x.copy())
                    synthetic_y.append(minority_class)
                break
                
            neighbor_indices = np.argsort(distances)[:k]
            
            # Generate synthetic samples
            # Distribute needed samples roughly equally among minority instances
            n_samples_for_this = n_synthetic_needed // n_minority
            if i < n_synthetic_needed % n_minority:
                n_samples_for_this += 1
                
            for _ in range(n_samples_for_this):
                neighbor_idx = rng.choice(neighbor_indices)
                neighbor_x = minority_X[neighbor_idx]
                
                new_x = np.empty_like(current_x)
                
                # Interpolate numerical
                gap = rng.random()
                if num_indices:
                    curr_num = current_x[num_indices].astype(float)
                    neigh_num = neighbor_x[num_indices].astype(float)
                    new_num = curr_num + gap * (neigh_num - curr_num)
                    new_x[num_indices] = new_num.astype(current_x.dtype)
                    
                # Categorical (mode/random)
                if cat_indices:
                    # Randomly pick from current or neighbor
                    for ci in cat_indices:
                        new_x[ci] = current_x[ci] if rng.random() > 0.5 else neighbor_x[ci]
                        
                synthetic_X.append(new_x)
                synthetic_y.append(minority_class)
                
        if not synthetic_X:
            return X, y
            
        synthetic_X = np.array(synthetic_X)
        synthetic_y = np.array(synthetic_y)
        
        # Combine
        final_X_arr = np.vstack([X_arr, synthetic_X])
        final_y_arr = np.concatenate([y_arr, synthetic_y])
        
        # Reconstruct DataFrame
        data = {}
        for idx, col in enumerate(X.columns):
            data[col] = final_X_arr[:, idx].astype(X.dtypes[col])
            
        final_X = DataFrame(data, columns=X.columns)
        
        # Reconstruct Series
        final_y = Series(final_y_arr, name=y.name)
        
        return final_X, final_y
