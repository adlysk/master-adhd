import numpy as np
from scipy.signal import welch

def calculate_hf_power(signal, fs=4, window_size=300, step_size=300):
    signal = list(signal.reshape(1, -1)[0])
    hf_powers = []
    num_samples = len(signal)

    for start in range(0, num_samples - window_size + 1, step_size):
        segment = signal[start:start + window_size]
        freqs, psd = welch(segment, fs, nperseg=window_size)
        hf_band = (freqs >= 0.15) & (freqs <= 0.4)
        hf_power = np.trapz(psd[hf_band], freqs[hf_band])
        hf_powers.append(hf_power)
        
    return hf_powers

def calculate_lf_hf_ratio(signal, fs=4, window_size=300, step_size=300):
    signal = list(signal.reshape(1, -1)[0])
    lf_powers = []
    hf_powers = []
    lf_hf_ratios = []
    num_samples = len(signal)
    
    for start in range(0, num_samples - window_size + 1, step_size):
        segment = signal[start:start + window_size]
        freqs, psd = welch(segment, fs, nperseg=window_size)
        
        lf_band = (freqs >= 0.04) & (freqs < 0.15)
        hf_band = (freqs >= 0.15) & (freqs < 0.4)
        
        lf_power = np.trapz(psd[lf_band], freqs[lf_band])
        hf_power = np.trapz(psd[hf_band], freqs[hf_band])
        
        lf_powers.append(lf_power)
        hf_powers.append(hf_power)
        
        if hf_power > 0:
            lf_hf_ratios.append(lf_power / hf_power)
        else:
            lf_hf_ratios.append(np.nan)

    return lf_powers, hf_powers, lf_hf_ratios

def hf_power_features(signal, fs, window_size, step_size):
    hf_powers = calculate_hf_power(signal, fs=fs, window_size=window_size, step_size=step_size)
    return hf_powers

def lf_hf_ratio_features(signal, fs, window_size, step_size):
    return calculate_lf_hf_ratio(signal, fs=fs, window_size=window_size, step_size=step_size)

def hf_max_feature(params, hf_powers):
    fs, window_size, step_size = params
    return {f"max_{fs}_{window_size}_{step_size}": np.max(hf_powers)}

def hf_min_feature(params, hf_powers):
    fs, window_size, step_size = params
    return {f"min_{fs}_{window_size}_{step_size}": np.min(hf_powers)}

def lf_std_feature(params, lf_hf):
    fs, window_size, step_size = params
    lf_powers, _, _ = lf_hf
    return {f"lf_std_{fs}_{window_size}_{step_size}": np.std(lf_powers)}

def lf_hf_ratio_median_feature(params, lf_hf):
    fs, window_size, step_size = params
    _, _, lf_hf_ratios = lf_hf
    return {f"lf_hf_ratio_median_{fs}_{window_size}_{step_size}": np.nanmedian(lf_hf_ratios)}

def lf_hf_ratio_cv_feature(params, lf_hf):
    fs, window_size, step_size = params
    _, _, lf_hf_ratios = lf_hf
    return {
        f"lf_hf_ratio_cv_{fs}_{window_size}_{step_size}": 
        np.nanstd(lf_hf_ratios) / np.nanmean(lf_hf_ratios) if np.nanmean(lf_hf_ratios) != 0 else np.nan
    }