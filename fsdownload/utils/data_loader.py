import pickle
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADJUST_DIR = PROJECT_ROOT / "fsdownload" / "adjust"
DATA_DIR = ADJUST_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = ADJUST_DIR / "artifacts"

INPUT_DIM_TEMP = 16
INPUT_DIM_PARAM = 27


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(device_arg="auto"):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_parent(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(obj, path):
    path = ensure_parent(path)
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def parse_pipe_rows(path):
    rows = []
    with open(path, "r") as handle:
        for raw in handle:
            parts = raw.strip().rstrip("|").split("|")
            if parts and parts[-1] == "":
                parts.pop()
            rows.append(list(map(float, parts)))
    return rows


def reorder_runtime_values(values):
    segment = values[32:36]
    tail = values[36:45]
    return values[:32] + tail + segment + values[45:]


def load_runtime_matrix(path):
    rows = [reorder_runtime_values(values) for values in parse_pipe_rows(path)]
    return np.asarray(rows, dtype=np.float32)


def runtime_to_training_matrix(runtime_matrix):
    return runtime_matrix[:, INPUT_DIM_TEMP:]


def split_training_matrix(training_matrix):
    return training_matrix[:, :INPUT_DIM_TEMP], training_matrix[:, INPUT_DIM_TEMP:]


def build_forecast_dataset(temp, param, window, pred_steps):
    temp_windows = []
    param_rows = []
    targets = []
    for index in range(len(temp) - window - pred_steps):
        temp_windows.append(temp[index:index + window])
        param_rows.append(param[index + window])
        delta_seq = temp[index + window + 1:index + window + 1 + pred_steps] - temp[index + window]
        targets.append(delta_seq)
    return (
        np.asarray(temp_windows, dtype=np.float32),
        np.asarray(param_rows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
    )


def to_device_tensors(device, *arrays):
    return tuple(torch.tensor(array, dtype=torch.float32, device=device) for array in arrays)


def extract_runtime_state(data):
    latest_row = data[-1]
    target_temp = latest_row[:INPUT_DIM_TEMP]
    current_temp = latest_row[INPUT_DIM_TEMP:INPUT_DIM_TEMP * 2]
    params = latest_row[INPUT_DIM_TEMP * 2:]
    return target_temp, current_temp, params


def runtime_temperature_history(data, window):
    return data[-(window + 1):-1, INPUT_DIM_TEMP:INPUT_DIM_TEMP * 2]


def list_raw_daily_files(raw_data_dir=RAW_DATA_DIR, date_from=None, date_to=None):
    raw_data_dir = Path(raw_data_dir or RAW_DATA_DIR)
    files = []
    for path in sorted(raw_data_dir.glob("*.txt")):
        stem = path.stem
        if not stem.isdigit() or len(stem) != 8:
            continue
        if date_from and stem < date_from:
            continue
        if date_to and stem > date_to:
            continue
        files.append(path)
    return files


def load_training_matrix_from_raw(raw_data_dir=RAW_DATA_DIR, date_from=None, date_to=None, explicit_files=None):
    if explicit_files:
        file_paths = [Path(path) for path in explicit_files]
    else:
        file_paths = list_raw_daily_files(raw_data_dir=raw_data_dir or RAW_DATA_DIR, date_from=date_from, date_to=date_to)

    if not file_paths:
        raise FileNotFoundError("No raw daily files matched the requested training range.")

    matrices = [runtime_to_training_matrix(load_runtime_matrix(path)) for path in file_paths]
    return np.concatenate(matrices, axis=0)


def resolve_runtime_data_path(raw_data_dir=RAW_DATA_DIR, today_path=None):
    if today_path:
        return Path(today_path)

    raw_data_dir = Path(raw_data_dir or RAW_DATA_DIR)
    candidate = raw_data_dir / "today_data.txt"
    if candidate.exists():
        return candidate

    daily_files = list_raw_daily_files(raw_data_dir=raw_data_dir)
    if not daily_files:
        raise FileNotFoundError("No runtime data file found in raw data directory.")
    return daily_files[-1]
