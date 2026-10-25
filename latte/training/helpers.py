import torch

def slice_time_series(x, n_windows):
    """
    Slice a batched time series (batch, time, channel) into N equal-size windows
    with minimal overlap along the time dimension.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch, time, channel).
    n_windows : int
        Number of windows to generate.

    Returns
    -------
    windows : torch.Tensor
        Output tensor of shape (batch, window_size * n_windows, channel).
    """
    if x.ndim != 3:
        raise ValueError("Input must have shape (batch, time, channel)")

    B, T, C = x.shape

    if n_windows <= 0 or n_windows > T:
        raise ValueError("n_windows must be between 1 and the time dimension size")

    # Window size = ceil(T / N) to cover the full time series
    win_size = (T + n_windows - 1) // n_windows

    # Stride so windows are evenly spread
    stride = (T - win_size) / max(n_windows - 1, 1)


    windows = []
    for i in range(n_windows):
        start = int(round(i * stride))
        end = start + win_size
        windows.append(x[:, start:end, :].unsqueeze(1))  # shape: (batch, window_size, channel)

    # Concatenate along the time dimension
    return torch.cat(windows, dim=1).reshape(B*n_windows, -1, C)


