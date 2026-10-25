import torch


class SubjectMaskedLoader:
    """Wraps a DataLoader yielding (x, x_sub, y) and rewrites x_sub.

    Selected samples are routed through slot `mask_value` (default 0, the
    reserved padding/unknown slot).

    Args:
        loader: underlying DataLoader yielding (x, x_sub, y) tuples.
        mask_prob: per-sample probability of overwriting x_sub with mask_value.
        force_mask: if True, every sample's x_sub becomes mask_value
                    (ignores mask_prob).
        mask_value: index to write into x_sub.
    """

    def __init__(self, loader, mask_prob=0.0, force_mask=False, mask_value=0):
        self.loader = loader
        self.mask_prob = float(mask_prob)
        self.force_mask = bool(force_mask)
        self.mask_value = int(mask_value)

    def set_mask_value(self, v):
        """Update the slot index that force_mask / mask_prob writes into x_sub."""
        self.mask_value = int(v)

    def __iter__(self):
        for batch in self.loader:
            if not isinstance(batch, (list, tuple)) or len(batch) != 3:
                yield batch
                continue
            x, sub, y = batch
            if self.force_mask:
                sub = torch.full_like(sub, self.mask_value)
            elif self.mask_prob > 0:
                pick = torch.rand(sub.shape, device=sub.device) < self.mask_prob
                if pick.any():
                    sub = torch.where(pick, torch.full_like(sub, self.mask_value), sub)
            yield x, sub, y

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset

    @property
    def batch_size(self):
        return self.loader.batch_size

    @property
    def batch_sampler(self):
        """Expose the underlying batch sampler.

        The trainer calls `batch_sampler.set_epoch(k)` on samplers that
        reshuffle per epoch (the domain-stratified one).
        """
        return getattr(self.loader, "batch_sampler", None)
