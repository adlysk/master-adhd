import os
import pandas as pd
from preprocess.file_processing import preprocess_files
from utils.utils import process_signals

def prepare_dataset(
    hrv_data_dir,
    data_dir,
    params_list,
    filter_fn,
    smooth_fn,
    normalize_fn
):
    df_patients = pd.read_csv(f"{data_dir}/patient_info.csv", sep=";")
    files = os.listdir(hrv_data_dir)

    signals, labels, times = preprocess_files(
        files, df_patients,
        filter_fn=filter_fn,
        smooth_fn=smooth_fn,
        normalize_fn=normalize_fn,
    )

    dataframes = []
    for feature_info in params_list:
        df_refactor = process_signals(
            signals, labels, feature_info["params"], feature_info["feature_func"], feature_info["feature_name"]
        )
        df_refactor['ADHD'] = labels
        dataframes.append(df_refactor)

    df = dataframes[0]
    for df_add in dataframes[1:]:
        df = df.merge(df_add, on=["id", "ADHD"], how="inner")

    df = df.drop(columns=["id"])
    return df
