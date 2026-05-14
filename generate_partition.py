import numpy as np
from sklearn.model_selection import train_test_split
from scipy.io import savemat

def run_partition_logic(X, y, groups):
    # 1. Stationarity Analysis
    unique_runs = np.unique(groups)
    cv_matrix = []
    for run in unique_runs:
        run_data = X[groups == run][:10] # Original segments only
        cv = (np.std(run_data, axis=0) / (np.abs(np.mean(run_data, axis=0)) + 1e-9)) * 100
        cv_results.append(cv)
    
    print(f"Stationarity Analysis Complete. Avg CV: {np.mean(cv_results):.2f}%")

    # 2. Condition-Based Split (85/15)
    unique_run_ids = np.unique(groups)
    # Get labels for each unique run
    _, first_indices = np.unique(groups, return_index=True)
    run_labels = y[first_indices]

    train_ids, test_ids = train_test_split(unique_run_ids, test_size=0.15, stratify=run_labels, random_state=3)
    
    idx_train = np.isin(groups, train_ids)
    idx_test = np.isin(groups, test_ids)

    # 3. Save for Script 3
    savemat('newResultsCL.mat', {'X': X, 'y': y, 'groups': groups, 'idx_train': idx_train, 'idx_test': idx_test})
