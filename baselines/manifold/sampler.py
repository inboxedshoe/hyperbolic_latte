import math
import warnings
from typing import List, Optional, Sequence

import numpy as np
from torch.utils.data.sampler import Sampler


class DomainStratifiedBatchSampler(Sampler):
    """Compose every batch as `domains_per_batch` x `obs_per_domain`.

    The batch size is ``D * N``. Domains are drawn
    without replacement within a batch and cycled through a reshuffled
    permutation across batches, so every domain is visited equally often
    regardless of how many trials it contributes. Within a domain, indices are
    drawn from a shuffled queue that is refilled when exhausted, so a small
    domain repeats within an epoch.

    Args:
        domain_labels:     per-sample domain id, length = len(dataset).
        domains_per_batch: D, number of distinct domains per batch. Clamped to
                           the number of domains actually present.
        obs_per_domain:    N, observations drawn from each chosen domain.
        num_batches:       batches per epoch. Defaults to
                           ``ceil(len(dataset) / (D * N))``.
        seed:              base RNG seed; the epoch counter is mixed in so
                           successive epochs differ but runs stay reproducible.
    """

    def __init__(
        self,
        domain_labels: Sequence[int],
        domains_per_batch: int = 5,
        obs_per_domain: int = 10,
        num_batches: Optional[int] = None,
        seed: int = 0,
    ):
        labels = np.asarray(domain_labels).reshape(-1)
        if labels.size == 0:
            raise ValueError("DomainStratifiedBatchSampler: empty domain_labels.")

        self.domains = np.unique(labels)
        self._pools = [np.flatnonzero(labels == d) for d in self.domains]

        n_domains = len(self.domains)
        self.obs_per_domain = int(obs_per_domain)
        self.domains_per_batch = int(min(domains_per_batch, n_domains))
        if self.domains_per_batch < domains_per_batch:
            warnings.warn(
                f"DomainStratifiedBatchSampler: domains_per_batch="
                f"{domains_per_batch} exceeds the {n_domains} domain(s) present; "
                f"using {self.domains_per_batch}. Effective batch size is now "
                f"{self.domains_per_batch * self.obs_per_domain}."
            )
        if self.domains_per_batch < 1 or self.obs_per_domain < 1:
            raise ValueError("domains_per_batch and obs_per_domain must be >= 1.")

        self.batch_size = self.domains_per_batch * self.obs_per_domain
        self.num_batches = (
            int(num_batches) if num_batches is not None
            else max(1, math.ceil(labels.size / self.batch_size))
        )
        self.seed = int(seed)
        self.epoch = 0

        small = [int(self.domains[i]) for i, p in enumerate(self._pools)
                 if len(p) < self.obs_per_domain]
        if small:
            warnings.warn(
                f"DomainStratifiedBatchSampler: domain(s) {small} have fewer "
                f"than obs_per_domain={self.obs_per_domain} samples; their "
                f"batch slots will contain repeats."
            )

    def set_epoch(self, epoch: int) -> None:
        """Reshuffle deterministically per epoch."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1315423911 * self.epoch)

        # Per-domain shuffled queues, refilled on exhaustion.
        queues: List[List[int]] = [
            list(rng.permutation(pool)) for pool in self._pools
        ]
        # Cycled domain permutation so every domain appears equally often.
        domain_order: List[int] = list(rng.permutation(len(self.domains)))

        for _ in range(self.num_batches):
            if len(domain_order) < self.domains_per_batch:
                domain_order.extend(rng.permutation(len(self.domains)))
            chosen, seen = [], set()
            while len(chosen) < self.domains_per_batch:
                d = domain_order.pop(0)
                if d in seen:                       # avoid a duplicate domain
                    domain_order.append(d)          # inside one batch
                    continue
                seen.add(d)
                chosen.append(d)

            batch: List[int] = []
            for d in chosen:
                q = queues[d]
                for _ in range(self.obs_per_domain):
                    if not q:
                        q.extend(rng.permutation(self._pools[d]))
                    batch.append(int(q.pop()))

            assert len(batch) == self.batch_size, (
                f"malformed batch: {len(batch)} != {self.batch_size}")
            yield batch
