import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from fsdownload.models.time_xer import TimeXerForecaster
from fsdownload.utils.data_loader import (
    ARTIFACTS_DIR,
    INPUT_DIM_PARAM,
    INPUT_DIM_TEMP,
    RAW_DATA_DIR,
    build_forecast_dataset,
    ensure_parent,
    get_device,
    load_training_matrix_from_raw,
    save_pickle,
    set_seed,
    split_training_matrix,
    to_device_tensors,
)


def train_model(args):
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    model_path = ensure_parent(args.model_path)
    scaler_temp_path = ensure_parent(args.scaler_temp_path)
    scaler_param_path = ensure_parent(args.scaler_param_path)
    scaler_delta_path = ensure_parent(args.scaler_delta_path)
    loss_plot_path = ensure_parent(args.loss_plot_path)

    training_matrix = load_training_matrix_from_raw(
        raw_data_dir=args.raw_data_dir,
        date_from=args.date_from,
        date_to=args.date_to,
        explicit_files=args.raw_files,
    )
    temp, param = split_training_matrix(training_matrix)
    temp_windows, param_rows, targets = build_forecast_dataset(temp, param, args.window_size, args.pred_steps)

    scaler_temp = StandardScaler().fit(temp)
    scaler_param = StandardScaler().fit(param)
    scaler_delta = StandardScaler().fit(targets.reshape(-1, INPUT_DIM_TEMP))

    temp_windows = scaler_temp.transform(temp_windows.reshape(-1, INPUT_DIM_TEMP)).reshape(
        -1, args.window_size, INPUT_DIM_TEMP
    )
    param_rows = scaler_param.transform(param_rows)
    targets = scaler_delta.transform(targets.reshape(-1, INPUT_DIM_TEMP)).reshape(
        -1, args.pred_steps, INPUT_DIM_TEMP
    )

    save_pickle(scaler_temp, scaler_temp_path)
    save_pickle(scaler_param, scaler_param_path)
    save_pickle(scaler_delta, scaler_delta_path)

    num_samples = len(temp_windows)
    train_size = int(args.train_ratio * num_samples)
    val_size = int(args.val_ratio * num_samples)

    x_temp_train = temp_windows[:train_size]
    x_temp_val = temp_windows[train_size:train_size + val_size]
    x_temp_test = temp_windows[train_size + val_size:]

    x_param_train = param_rows[:train_size]
    x_param_val = param_rows[train_size:train_size + val_size]
    x_param_test = param_rows[train_size + val_size:]

    y_train = targets[:train_size]
    y_val = targets[train_size:train_size + val_size]
    y_test = targets[train_size + val_size:]

    x_temp_train, x_param_train, y_train = to_device_tensors(device, x_temp_train, x_param_train, y_train)
    x_temp_val, x_param_val, y_val = to_device_tensors(device, x_temp_val, x_param_val, y_val)
    x_temp_test, x_param_test, y_test = to_device_tensors(device, x_temp_test, x_param_test, y_test)

    train_loader = DataLoader(TensorDataset(x_temp_train, x_param_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_temp_val, x_param_val, y_val), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_temp_test, x_param_test, y_test), batch_size=args.batch_size, shuffle=False)

    model = TimeXerForecaster(
        input_dim_temp=args.input_dim_temp,
        input_dim_param=args.input_dim_param,
        pred_steps=args.pred_steps,
        patch_len=args.patch_len,
        stride=args.stride,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_lambda)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    epochs_without_improvement = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0

        for x_temp_batch, x_param_batch, y_batch in train_loader:
            noise_temp = torch.randn_like(x_temp_batch) * args.noise_level
            noise_param = torch.randn_like(x_param_batch) * args.noise_level

            predictions = model(x_temp_batch + noise_temp, x_param_batch + noise_param)
            loss = loss_fn(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        train_loss = train_loss_sum / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x_temp_batch, x_param_batch, y_batch in val_loader:
                predictions = model(x_temp_batch, x_param_batch)
                val_loss_sum += loss_fn(predictions, y_batch).item()

        val_loss = val_loss_sum / len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {args.patience} stagnant epochs.")
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    test_loss_sum = 0.0
    with torch.no_grad():
        for x_temp_batch, x_param_batch, y_batch in test_loader:
            predictions = model(x_temp_batch, x_param_batch)
            test_loss_sum += loss_fn(predictions, y_batch).item()
    print(f"Test Loss: {test_loss_sum / len(test_loader):.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TimeXer Training Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(loss_plot_path)
    plt.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Train the TimeXer temperature forecaster.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--pred_steps", type=int, default=200)
    parser.add_argument("--window_size", type=int, default=100)

    parser.add_argument("--input_dim_temp", type=int, default=INPUT_DIM_TEMP)
    parser.add_argument("--input_dim_param", type=int, default=INPUT_DIM_PARAM)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)

    parser.add_argument("--raw_data_dir", type=str, default=str(RAW_DATA_DIR))
    parser.add_argument("--date_from", type=str, default=None)
    parser.add_argument("--date_to", type=str, default=None)
    parser.add_argument("--raw_files", nargs="*", default=None)
    parser.add_argument("--model_path", type=str, default=str(ARTIFACTS_DIR / "model.pth"))
    parser.add_argument("--scaler_temp_path", type=str, default=str(ARTIFACTS_DIR / "scaler_temp.pkl"))
    parser.add_argument("--scaler_param_path", type=str, default=str(ARTIFACTS_DIR / "scaler_param.pkl"))
    parser.add_argument("--scaler_delta_path", type=str, default=str(ARTIFACTS_DIR / "scaler_delta.pkl"))
    parser.add_argument("--loss_plot_path", type=str, default=str(ARTIFACTS_DIR / "training_loss.png"))

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--l2_lambda", type=float, default=1e-3)
    parser.add_argument("--noise_level", type=float, default=0.1)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
