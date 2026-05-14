import numpy as np
import os
from scipy.io import loadmat
from scipy.signal import medfilt
from scipy.fft import fft, fftfreq
from scipy.stats import skew

def build_feature_matrix(num_conditions, condition_labels, data_folder, fs=400000):
    num_aug = 2
    num_segments = 10
    X, y, groups = [], [], []

    for cond in range(1, num_conditions + 1):
        file_path = os.path.join(data_folder, f"{cond}.mat")
        if not os.path.exists(file_path): continue

        # Load signal
        mat_data = loadmat(file_path)
        signal_orig = mat_data[list(mat_data.keys())[-1]].flatten().astype(float)

        for a in range(1, num_aug + 1):
            if a == 1:
                signal = signal_orig
            else:
                noise = 0.01 * (np.ptp(signal_orig)) * np.random.randn(*signal_orig.shape)
                signal = signal_orig + noise

            # 1. Filtering & Normalization
            raw_smooth = medfilt(signal, 5)
            v_water, v_air = np.percentile(raw_smooth, [1, 99])
            norm_signal = (raw_smooth - v_air) / (v_water - v_air + 1e-9)

            # 2. Interface Detection (Refine Interfaces Logic)
            grad1 = np.gradient(norm_signal)
            grad2 = np.gradient(grad1)
            is_liquid = (norm_signal > 0.6).astype(int)
            trans = np.diff(np.concatenate(([0], is_liquid, [0])))
            starts = np.where(trans == 1)[0]
            ends = np.where(trans == -1)[0] - 1

            binary_signal = np.zeros_like(norm_signal)
            for s_c, e_c in zip(starts, ends):
                if (e_c - s_c) < 3: continue
                # Front refinement (2nd order grad)
                f_w = np.arange(s_c, s_c + int(0.2 * (e_c - s_c)) + 1)
                t_s = f_w[np.where(grad2[f_w] >= 0.5 * np.max(grad2[f_w]))[0][0]]
                # Rear refinement (1st order grad)
                r_w = np.arange(s_c + int(0.5 * (e_c - s_c)), e_c + 1)
                t_e = r_w[np.where(grad1[r_w] >= 0.1 * np.max(grad1[r_w]))[0][0]]
                binary_signal[t_s:t_e+1] = 1

            # 3. Feature Extraction per Segment
            N = len(binary_signal)
            for k in range(num_segments):
                s_idx, e_idx = int(k*N/num_segments), int((k+1)*N/num_segments)
                seg = binary_signal[s_idx:e_idx]
                
                # Local events
                mask = (starts >= s_idx) & (ends < e_idx)
                seg_starts = starts[mask] - s_idx
                seg_ends = ends[mask] - s_idx
                durations = seg_ends - seg_starts + 1

                # Feature: Entropy
                counts, _ = np.histogram(seg_starts, bins=5, range=(0, len(seg)))
                p = counts / (np.sum(counts) + 1e-9)
                f_entropy = -np.sum(p * np.log2(p + 1e-9))

                # Feature: Event Rate
                f_eventRate = len(seg_starts) / (len(seg)/fs)

                # Feature: CV of IAT
                iat = np.diff(seg_starts)
                f_cv_iat = np.std(iat)/(np.mean(iat)+1e-9) if len(iat) > 1 else 0

                # Feature: Sigma Pulse
                f_sigmaPulse = np.std(durations) if len(durations) > 1 else 0

                # Feature: Centroid
                Y = np.abs(fft(seg - np.mean(seg)))
                p1 = Y[:len(seg)//2+1]
                freqs = np.linspace(0, fs/2, len(p1))
                f_centroid = np.sum(freqs * p1) / (np.sum(p1) + 1e-9)

                # Feature: Max Gas Gap
                if len(seg_starts) > 1:
                    gaps = seg_starts[1:] - seg_ends[:-1]
                    f_maxGasGap = np.max(gaps) / fs
                else:
                    f_maxGasGap = (len(seg)/fs) if len(seg_starts) == 0 else 0

                X.append([f_entropy, f_eventRate, f_cv_iat, f_sigmaPulse, f_centroid, f_maxGasGap])
                y.append(condition_labels[cond-1])
                groups.append(cond)

    return np.array(X), np.array(y), np.array(groups)
