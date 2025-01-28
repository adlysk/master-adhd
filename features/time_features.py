import numpy as np

def calculate_rmssd(nn_intervals):
    diffs = np.diff(nn_intervals)
    if len(diffs) < 2:
        return np.nan
    return np.sqrt(np.mean(diffs**2))

def calculate_sdnn(nn_intervals):
    if len(nn_intervals) < 2:
        return np.nan
    return np.std(nn_intervals)

def calculate_time_domain_features(signal, window_size=300, step_size=300):
    signal = list(signal.reshape(1, -1)[0])
    rmssd_values = []
    sdnn_values = []
    num_samples = len(signal)
    for start in range(0, num_samples - window_size + 1, step_size):
        window = signal[start:start + window_size]
        if len(set(window)) == 1:
            continue
        rmssd = calculate_rmssd(window)
        sdnn = calculate_sdnn(window)
        if not np.isnan(rmssd):
            rmssd_values.append(rmssd)
        if not np.isnan(sdnn):
            sdnn_values.append(sdnn)
    return rmssd_values, sdnn_values

def time_domain_features(signal, window_size, step_size):
    return calculate_time_domain_features(signal, window_size=window_size, step_size=step_size)