import numpy as np

def calculate_poincare_features(nn_intervals):
    diffs = np.diff(nn_intervals)
    sd1 = np.sqrt(np.var(diffs) / 2)
    sd2 = np.sqrt(2 * np.var(nn_intervals) - np.var(diffs) / 2)
    return sd1, sd2, sd1 / sd2

def calculate_geometric_features(nn_intervals):
    if len(nn_intervals) < 2:
        return np.nan, np.nan
    hist, bin_edges = np.histogram(nn_intervals, bins=50)
    max_bin = np.max(hist)
    triangular_index = len(nn_intervals) / max_bin
    tinn = bin_edges[-1] - bin_edges[0]
    return triangular_index, tinn

def tinn_feature(params, time_features):
    window_size, step_size = params
    nn_vals, _ = time_features
    _, tinn = calculate_geometric_features(nn_vals)
    return {f"tinn_{window_size}_{step_size}": tinn}

def poincare_sd1_sd2_ratio_feature(params, time_features):
    window_size, step_size = params
    nn_vals, _ = time_features
    _, _, sd1_sd2_ratio = calculate_poincare_features(nn_vals)
    return {f"poincare_sd1_sd2_ratio_{window_size}_{step_size}": sd1_sd2_ratio}