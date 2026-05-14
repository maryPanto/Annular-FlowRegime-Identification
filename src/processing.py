import numpy as np
import os
from scipy.io import loadmat
from scipy.signal import medfilt
from scipy.fft import fft

def build_feature_matrix(num_conditions, condition_labels, data_folder, fs=400000):
    num_aug = 2
    num_segments = 10
    X, y, groups = [], [], []

    print(f"Starting Feature Extraction for {num_conditions} conditions...")
    for cond in range(1, num_conditions + 1):
        file_path = os.path.join(data_folder, f"{cond}.mat")
        if not os.path.exists(file_path): continue

        mat_data = loadmat(file_path)
        var_name = list(mat_data.keys())[-1]  # Get the actual data variable
        signal_orig = mat_data[var_name].flatten().astype(float)

        for a in range(1, num_aug + 1):
            if a == 1:
                signal = signal_orig
            else:
                noise_level = 0.01 * (np.max(signal_orig) - np.min(signal_orig))
                signal = signal_orig + noise_level * np.random.randn(*signal_orig.shape)

            # 1. Median Filtering & Normalization
            raw_smooth = medfilt(signal, kernel_size=5)
            v_water_base = np.percentile(raw_smooth, 1)
            v_air_base = np.percentile(raw_smooth, 99)
            norm_signal = (raw_smooth - v_air_base) / (v_water_base - v_air_base + 1e-9)

            # 2. Gradient Calculation
            grad1 = np.gradient(norm_signal)
            grad2 = np.gradient(grad1)

            # 3. Interface Refinement
            is_liquid = (norm_signal > 0.6).astype(int)
            transitions = np.diff(np.concatenate(([0], is_liquid, [0])))
            start_indices = np.where(transitions == 1)[0]
            end_indices = np.where(transitions == -1)[0] - 1

            binary_signal = np.zeros_like(norm_signal)
            for s_cand, e_cand in zip(start_indices, end_indices):
                if (e_cand - s_cand) < 3: continue
                # Front (2nd order grad) & Rear (1st order grad)
                binary_signal[s_cand:e_cand+1] = 1 # Simplified for logic matching

            # 4. Segmentation & Feature Extraction
            N = len(binary_signal)
            for k in range(num_segments):
                start_idx = int(k * N / num_segments)
                end_idx = int((k + 1) * N / num_segments)
                seg = binary_signal[start_idx:end_idx]
                
                # Feature Logic (Entropy, EventRate, CV_IAT, SigmaPulse, Centroid, MaxGasGap)
                mask = (start_indices >= start_idx) & (end_indices < end_idx)
                seg_starts = start_indices[mask] - start_idx
                num_events = len(seg_starts)
                
                f_eventRate = num_events / (len(seg) / fs)
                f_dtf = np.sum(seg == 1) / len(seg)
                
                # Spectral Centroid
                Y = np.abs(fft(seg - np.mean(seg)))
                P1 = Y[:len(seg)//2+1]
                freqs = np.linspace(0, fs/2, len(P1))
                f_centroid = np.sum(freqs * P1) / (np.sum(P1) + 1e-9)

                # Entropy Calculation
                counts, _ = np.histogram(seg_starts, bins=5, range=(0, len(seg)))
                p = counts / (np.sum(counts) + 1e-9)
                f_entropy = -np.sum(p * np.log2(p + 1e-9))

                X.append([f_entropy, f_eventRate, 0, 0, f_centroid, 0]) # Placeholder for complexity
                y.append(condition_labels[cond-1])
                groups.append(cond)

    return np.array(X), np.array(y), np.array(groups)
