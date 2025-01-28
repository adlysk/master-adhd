def default_filter_fn(df):
    return df[(df.HRV > 250) & (df.HRV < 2000)]

def no_filter_fn(df):
    return df

def rolling_mean_smooth_fn(hrv_series):
    return hrv_series.rolling(window=5, center=True, min_periods=1).mean()

def no_smooth_fn(hrv_series):
    return hrv_series