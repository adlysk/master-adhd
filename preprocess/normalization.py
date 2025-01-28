from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def z_score_normalize_fn(hrv_series):
    return (hrv_series - hrv_series.mean()) / hrv_series.std()

def min_max_scaler_fn(hrv_series):
    scaler = MinMaxScaler()
    return scaler.fit_transform(hrv_series.values.reshape(-1, 1)).flatten()

def robust_scaler_fn(hrv_series):
    scaler = RobustScaler()
    return scaler.fit_transform(hrv_series.values.reshape(-1, 1)).flatten()

def standard_scaler_fn(hrv_series):
    scaler = StandardScaler()
    return scaler.fit_transform(hrv_series.values.reshape(-1, 1)).flatten()

def no_normalize_fn(hrv_series):
    return hrv_series