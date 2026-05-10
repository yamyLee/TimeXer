import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from fsdownload.models.time_xer import TimeXerForecaster
from fsdownload.utils.data_loader import (
    ARTIFACTS_DIR,
    INPUT_DIM_PARAM,
    INPUT_DIM_TEMP,
    RAW_DATA_DIR,
    extract_runtime_state,
    get_device,
    load_pickle,
    load_runtime_matrix,
    resolve_runtime_data_path,
    runtime_temperature_history,
    set_seed,
)


def write_pipe_values(values, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.array(values, dtype=float)
    clipped[clipped < 0] = 0
    with open(path, "w") as handle:
        handle.write("|".join(f"{value:.2f}" for value in clipped))


def append_suggestion_log(values, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = "|".join(f"{value:.2f}" for value in values) + f"|{timestamp}"
    with open(path, "a") as handle:
        handle.write(log_line + "\n")


def append_feedback_log(current_params, suggestion_path, feedback_log_path):
    suggestion_path = Path(suggestion_path)
    if not suggestion_path.exists():
        return

    content = suggestion_path.read_text().strip()
    if not content:
        return

    suggested_params = np.array(list(map(float, content.split("|"))), dtype=np.float32)
    if len(suggested_params) != len(current_params):
        return

    errors = np.abs(current_params - suggested_params)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    feedback_log_path = Path(feedback_log_path)
    feedback_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feedback_log_path, "a") as handle:
        handle.write("|".join(f"{value:.4f}" for value in errors) + f"|{timestamp}\n")


def suggest_adjustment(args):
    set_seed(args.seed)
    device = get_device(args.device)

    model_path = Path(args.model_path)
    scaler_temp_path = Path(args.scaler_temp_path)
    scaler_param_path = Path(args.scaler_param_path)
    scaler_delta_path = Path(args.scaler_delta_path)

    missing = [path for path in [model_path, scaler_temp_path, scaler_param_path, scaler_delta_path] if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing artifacts: {missing_list}")

    scaler_temp = load_pickle(scaler_temp_path)
    scaler_param = load_pickle(scaler_param_path)
    scaler_delta = load_pickle(scaler_delta_path)

    model = TimeXerForecaster(
        input_dim_temp=INPUT_DIM_TEMP,
        input_dim_param=INPUT_DIM_PARAM,
        pred_steps=args.pred_steps,
        patch_len=args.patch_len,
        stride=args.stride,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    runtime_data_path = resolve_runtime_data_path(raw_data_dir=args.raw_data_dir, today_path=args.today_path)
    runtime_data = load_runtime_matrix(runtime_data_path)
    if len(runtime_data) < args.window_size + 1:
        raise ValueError("Not enough runtime rows to build an input window.")

    target_temp, current_temp, current_params = extract_runtime_state(runtime_data)
    append_feedback_log(current_params, args.output_path, args.feedback_log_path)

    deviation = current_temp - target_temp
    mask = np.abs(deviation) > args.temperature_tolerance
    if not np.any(mask):
        print("No adjustment required.")
        write_pipe_values(current_params, args.output_path)
        append_suggestion_log(current_params, args.output_log_path)
        return

    history = runtime_temperature_history(runtime_data, args.window_size)
    temp_window = scaler_temp.transform(history).reshape(1, args.window_size, INPUT_DIM_TEMP)
    param_vector = scaler_param.transform(np.asarray(current_params).reshape(1, -1))[0]

    temp_tensor = torch.tensor(temp_window, dtype=torch.float32, device=device)
    param_tensor = torch.tensor(param_vector, dtype=torch.float32, device=device, requires_grad=True)
    current_temp_tensor = torch.tensor(current_temp, dtype=torch.float32, device=device)
    target_temp_tensor = torch.tensor(target_temp, dtype=torch.float32, device=device)
    original_param_tensor = param_tensor.detach().clone()

    delta_scale = torch.tensor(scaler_delta.scale_, dtype=torch.float32, device=device)
    delta_mean = torch.tensor(scaler_delta.mean_, dtype=torch.float32, device=device)

    optimizer = Adam([param_tensor], lr=args.optim_lr)
    loss_fn = nn.SmoothL1Loss()
    model.train()

    best_loss = float("inf")
    best_param = None

    for _ in range(args.optim_steps):
        optimizer.zero_grad()
        predicted_delta = model(temp_tensor, param_tensor.unsqueeze(0))[0]
        predicted_temp = current_temp_tensor.unsqueeze(0) + predicted_delta * delta_scale + delta_mean

        temp_loss = loss_fn(predicted_temp[-args.mean_window:].mean(dim=0)[mask], target_temp_tensor[mask])
        reg_loss = torch.mean((param_tensor - original_param_tensor) ** 2)
        loss = temp_loss + args.regularization_weight * reg_loss
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_param = param_tensor.detach().cpu().numpy().copy()

    if best_param is None:
        raise RuntimeError("Adjustment optimization failed to produce a candidate parameter vector.")

    suggestion = scaler_param.inverse_transform(best_param.reshape(1, -1))[0]
    write_pipe_values(suggestion, args.output_path)
    append_suggestion_log(suggestion, args.output_log_path)

    print("Suggested parameter updates:")
    for index, (old_value, new_value) in enumerate(zip(current_params, suggestion), start=1):
        print(f"param_{index:02d}: {new_value - old_value:+.2f} ({old_value:.2f} -> {new_value:.2f})")


def build_parser():
    parser = argparse.ArgumentParser(description="Optimize control parameters with the trained TimeXer model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--pred_steps", type=int, default=200)
    parser.add_argument("--mean_window", type=int, default=10)
    parser.add_argument("--temperature_tolerance", type=float, default=2.0)
    parser.add_argument("--optim_steps", type=int, default=300)
    parser.add_argument("--optim_lr", type=float, default=0.02)
    parser.add_argument("--regularization_weight", type=float, default=0.0)

    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--raw_data_dir", type=str, default=str(RAW_DATA_DIR))
    parser.add_argument("--today_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=str(ARTIFACTS_DIR / "model.pth"))
    parser.add_argument("--scaler_temp_path", type=str, default=str(ARTIFACTS_DIR / "scaler_temp.pkl"))
    parser.add_argument("--scaler_param_path", type=str, default=str(ARTIFACTS_DIR / "scaler_param.pkl"))
    parser.add_argument("--scaler_delta_path", type=str, default=str(ARTIFACTS_DIR / "scaler_delta.pkl"))
    parser.add_argument("--output_path", type=str, default=str(ARTIFACTS_DIR / "suggested_params.txt"))
    parser.add_argument("--output_log_path", type=str, default=str(ARTIFACTS_DIR / "suggested_params_log.txt"))
    parser.add_argument("--feedback_log_path", type=str, default=str(ARTIFACTS_DIR / "adjustment_feedback_log.txt"))
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    suggest_adjustment(args)


if __name__ == "__main__":
    main()
