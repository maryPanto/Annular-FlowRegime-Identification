import numpy as np
from src.processing import build_feature_matrix
from sklearn.model_selection import train_test_split
from scipy.io import loadmat, savemat

# 1. Build the Matrix
labels = loadmat('condition_labels.mat')['condition_labels'].flatten()
X, y, groups = build_feature_matrix(46, labels, 'Final Data 2')

# 2. Stationarity Analysis
unique_runs = np.unique(groups)
cv_matrix = []
for run in unique_runs:
    run_data = X[groups == run][:10] # Original segments
    cv = (np.std(run_data, axis=0) / (np.abs(np.mean(run_data, axis=0)) + 1e-9)) * 100
    cv_matrix.append(cv)
print(f"Average CV: {np.mean(cv_matrix, axis=0)}")

# 3. Condition-Based Split (85/15)
unique_ids = np.unique(groups)
_, first_idx = np.unique(groups, return_index=True)
run_labels = y[first_idx]

train_ids, test_ids = train_test_split(unique_ids, test_size=0.15, stratify=run_labels, random_state=3)

idxOuterTrain = np.isin(groups, train_ids)
idxOuterTest = np.isin(groups, test_ids)

# Save result
savemat('newResultsCL.mat', {'X':X, 'y':y, 'groups':groups, 'idxOuterTrain':idxOuterTrain, 'idxOuterTest':idxOuterTest})
