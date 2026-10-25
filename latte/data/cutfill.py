import torch


class CutFillConfig:
    __slots__ = ("min_frac", "max_frac", "fill_value", "loss_on_full")
    def __init__(self, min_frac=0.1, max_frac=0.25, fill_value=0.0, loss_on_full=False):
        self.min_frac = float(min_frac)
        self.max_frac = float(max_frac)
        self.fill_value = fill_value  # "mean", float, or channel-wise Tensor
        self.loss_on_full = bool(loss_on_full)
    def __repr__(self):
        return (f"CutFillConfig(min_frac={self.min_frac}, max_frac={self.max_frac}, "
                f"fill_value={self.fill_value}, loss_on_full={self.loss_on_full})")


def _make_filler(x_b, fill_value, device, dtype, C):
    """Return a (1, C, 1) filler for one sample."""
    if fill_value == "mean":
        return x_b.mean(dim=-1, keepdim=True)  # (1, C, 1)
    if isinstance(fill_value, torch.Tensor):
        f = fill_value.to(device=device, dtype=dtype).view(1, -1, 1)
        if f.size(1) not in (1, C):
            raise ValueError(f"fill_value Tensor must have 1 or {C} channels")
        return f
    return torch.as_tensor(fill_value, device=device, dtype=dtype).view(1, 1, 1)


def _rand_span(T, min_frac, max_frac):
    min_len = max(1, int(T * min_frac))
    max_len = max(min_len, int(T * max_frac))
    span_len = torch.randint(min_len, max_len + 1, ()).item()
    start = torch.randint(0, max(1, T - span_len + 1), ()).item()
    return start, start + span_len


def apply_cut_and_fill(x: torch.Tensor, cfg: CutFillConfig):
    """
    EEG cut-and-fill for x shaped (B, 1, C, T).
      - Masks one contiguous time span per batch item, across all channels.
      - Fills with cfg.fill_value (scalar or Tensor broadcastable to (1,1,C,1)).

    Returns:
      x_masked: (B, 1, C, T)
      mask:     (B, 1, C, T) boolean (True where masked)
      spans:    list of (start, end) for each item
    """

    assert x.ndim == 4 and x.size(1) == 1, "Expected x with shape (B, 1, C, T)."
    B, _, C, T = x.shape
    device, dtype = x.device, x.dtype

    x_masked = x.clone()
    mask = torch.zeros((B, 1, C, T), dtype=torch.bool, device=device)
    spans = []

    for b in range(B):
        s, e = _rand_span(T, cfg.min_frac, cfg.max_frac)
        spans.append((s, e))
        filler = _make_filler(x[b, 0], cfg.fill_value, device, dtype, C)  # (C, 1)
        x_masked[b, 0, :, s:e] = filler
        mask[b, :, :, s:e] = True

    return x_masked, mask, spans


