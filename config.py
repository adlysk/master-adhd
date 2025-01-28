from features.lf_hf_features import (
    hf_power_features,
    lf_hf_ratio_features,
    hf_max_feature,
    hf_min_feature,
    lf_std_feature,
    lf_hf_ratio_median_feature,
    lf_hf_ratio_cv_feature,
)
from features.time_features import time_domain_features
from features.geometric_features import tinn_feature, poincare_sd1_sd2_ratio_feature
from preprocess.filtering import default_filter_fn, rolling_mean_smooth_fn
from preprocess.normalization import z_score_normalize_fn

PARAMS_LIST = [
    {"params": [(1, 300, 300)], "feature_func": hf_power_features, "feature_name": hf_max_feature},
    {"params": [(4, 600, 150)], "feature_func": hf_power_features, "feature_name": hf_min_feature},
    {"params": [(8, 4000, 2000)], "feature_func": lf_hf_ratio_features, "feature_name": lf_std_feature},
    {"params": [(8, 4000, 2000)], "feature_func": lf_hf_ratio_features, "feature_name": lf_hf_ratio_median_feature},
    {"params": [(2, 150, 50)], "feature_func": lf_hf_ratio_features, "feature_name": lf_hf_ratio_cv_feature},
    {"params": [(180, 60)], "feature_func": time_domain_features, "feature_name": tinn_feature},
    {"params": [(400, 100)], "feature_func": time_domain_features, "feature_name": poincare_sd1_sd2_ratio_feature},
]

DEFAULT_FILTER_FN = default_filter_fn
DEFAULT_SMOOTH_FN = rolling_mean_smooth_fn
DEFAULT_NORMALIZE_FN = z_score_normalize_fn

DATA_DIR = 'data'
HRV_DATA_DIR = f"{DATA_DIR}/hrv_data"
