import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

# Load the clean modeling feature matrix
df = pd.read_csv('/a0/bitcore/workspace/data/features/clean_modeling_feature_matrix.csv')

# Separate features from metadata columns
metadata_cols = ['antibody_id', 'fold']
feature_cols = [col for col in df.columns if col not in metadata_cols]

# Filter out non-numeric columns
numeric_df = df[feature_cols].select_dtypes(include=[np.number])
numeric_cols = numeric_df.columns.tolist()

print(f"Total original features: {len(feature_cols)}")
print(f"Numeric features: {len(numeric_cols)}")

# Identify different types of features among numeric features
# CDR features (starting with cdr_ but not ending with _seq)
cdr_cols = [col for col in numeric_cols if col.startswith('cdr_') and not col.endswith('_seq')]

# MI features (starting with mi_)
mi_cols = [col for col in numeric_cols if col.startswith('mi_')]

# Isotype features
isotype_cols = [col for col in numeric_cols if 'isotype' in col or col.startswith('is_')]

# Select exactly 6 isotype features (first 6)
isotype_cols = isotype_cols[:6]

# Other features (all remaining numeric features)
other_cols = [col for col in numeric_cols if col not in cdr_cols and col not in mi_cols and col not in isotype_cols]

print(f"\nCDR features: {len(cdr_cols)}")
print(f"MI features: {len(mi_cols)}")
print(f"Isotype features (selected): {len(isotype_cols)}")
print(f"Other features: {len(other_cols)}")

# 1. Apply PCA to CDR features (reduce to 20 principal components)
cdr_data = df[cdr_cols].values

# Standardize the CDR features
scaler = StandardScaler()
cdr_scaled = scaler.fit_transform(cdr_data)

# Apply PCA
pca = PCA(n_components=20)
cdr_pca = pca.fit_transform(cdr_scaled)

# Create DataFrame with PCA components
cdr_pca_df = pd.DataFrame(cdr_pca, columns=[f'cdr_pc_{i+1}' for i in range(20)])

print(f"\nCDR features reduced to {cdr_pca_df.shape[1]} principal components")

# 2. Select MI features (6 most informative mutual information features)
# For now, we'll use all 6 MI features as they are already selected
mi_df = df[mi_cols].copy()

print(f"\nMI features selected: {len(mi_cols)}")

# 3. Include isotype features (exactly 6 isotype features)
isotype_df = df[isotype_cols].copy()

print(f"\nIsotype features included: {len(isotype_cols)}")

# 4. Select other features (55 features based on variance)
other_data = df[other_cols].values

# Apply variance threshold to select top 55 features
selector = VarianceThreshold()
other_selected = selector.fit_transform(other_data)

# If we have more than 55 features, select top 55 based on variance
if other_selected.shape[1] > 55:
    # Calculate variances
    variances = np.var(other_selected, axis=0)
    # Get indices of top 55 features
    top_indices = np.argpartition(variances, -55)[-55:]
    # Select top 55 features
    other_selected = other_selected[:, top_indices]
    # Get corresponding column names
    other_cols_selected = [other_cols[i] for i in top_indices]
else:
    other_cols_selected = other_cols

other_df = pd.DataFrame(other_selected, columns=other_cols_selected)

print(f"\nOther features selected: {other_df.shape[1]}")

# 5. Combine all selected features
final_df = pd.concat([df[['antibody_id']], cdr_pca_df, mi_df, isotype_df, other_df], axis=1)

# 6. Validate feature set (ensure exactly 87 features)
print(f"\nFinal feature matrix shape: {final_df.shape}")
print(f"Features (excluding antibody_id): {final_df.shape[1] - 1}")

# Save the reduced feature matrix
final_df.to_csv('/a0/bitcore/workspace/data/features/reduced_modeling_feature_matrix_87.csv', index=False)

print(f"\nReduced feature matrix saved to /a0/bitcore/workspace/data/features/reduced_modeling_feature_matrix_87.csv")
