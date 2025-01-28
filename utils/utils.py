from tqdm import tqdm
import pandas as pd

def process_signals(signals, labels, params, feature_func, feature_name):
    data = []
    for idx, signal in tqdm(enumerate(signals), desc=f"Processing {feature_name.__name__}"):
        signal_stats = {"id": idx}
        for param_set in params:
            feature_values = feature_func(signal, *param_set)
            if feature_values:
                signal_stats.update(feature_name(param_set, feature_values))
        data.append(signal_stats)
    return pd.DataFrame(data)