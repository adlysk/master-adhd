import pandas as pd
from config import HRV_DATA_DIR

def preprocess_file(file, df_patients, filter_fn, smooth_fn, normalize_fn):
    """
    Preprocess a single file to extract signals, labels, and times with customizable steps.
    
    Parameters:
    - file: str, file name
    - df_patients: pd.DataFrame, patient information with ADHD labels
    - filter_fn: function, function to filter HRV values
    - smooth_fn: function, function to smooth HRV values
    - normalize_fn: function, function to normalize HRV values
    
    Returns:
    - signal: np.array, preprocessed HRV signal
    - label: int, ADHD label for the patient
    - time: list, time since start for each HRV value
    """
    if file == 'patient_info.csv':
        return None, None, None

    df = pd.read_csv(f'{HRV_DATA_DIR}/{file}', sep=';')
    df = filter_fn(df)
    df['HRV_Smoothed'] = smooth_fn(df['HRV'])
    ID = file.split('.')[0].split('_')[-1]
    ADHD = df_patients[df_patients.ID == int(ID)]['ADHD'].values[0]
    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"])
    df["TimeSinceStart"] = (df["TIMESTAMP"] - df["TIMESTAMP"].iloc[0]).dt.total_seconds()
    signal = normalize_fn(df['HRV_Smoothed'])
    if isinstance(signal, pd.Series):
        signal = signal.to_numpy()
    signal = signal.reshape(-1, 1)

    return signal, ADHD, df['TimeSinceStart'].values.tolist()



def preprocess_files(files, df_patients, filter_fn, smooth_fn, normalize_fn):
    """
    Preprocess multiple files to extract signals, labels, and times with customizable steps.
    
    Parameters:
    - files: list, list of file names
    - df_patients: pd.DataFrame, patient information with ADHD labels
    - filter_fn: function, function to filter HRV values
    - smooth_fn: function, function to smooth HRV values
    - normalize_fn: function, function to normalize HRV values
    
    Returns:
    - signals: list of np.array, preprocessed HRV signals
    - labels: list of int, ADHD labels for each patient
    - times: list of list, times since start for each HRV value
    """
    signals, labels, times = [], [], []

    for file in files:
        signal, label, time = preprocess_file(file, df_patients, filter_fn, smooth_fn, normalize_fn)
        if signal is not None:
            signals.append(signal)
            labels.append(label)
            times.append(time)

    return signals, labels, times