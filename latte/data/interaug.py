import numpy as np
import torch


def interaug(batch, subject_ids=None):
    x, y = batch
    new_samples = torch.zeros_like(x)
    new_labels = torch.zeros_like(y)
    if subject_ids is not None:
        new_subject_ids = torch.zeros_like(subject_ids)
    current = 0
    # largest divisor of the time length in 5-10 to split it into equal chunks
    time_dim = new_samples.shape[-1]
    n_chunks = 8  # default
    for nc in range(10, 4, -1):  # try 10, 9, 8, 7, 6, 5
        if time_dim % nc == 0:
            n_chunks = nc
            break
    for cls in torch.unique(y):
        cls_mask = y == cls
        x_cls = x[cls_mask]
        if subject_ids is not None:
            sub_cls = subject_ids[cls_mask]
        chunks = torch.cat(torch.chunk(x_cls, chunks=n_chunks, dim=-1))
        # indices = np.random.choice(len(x_cls), size=(len(x_cls), n_chunks),
        #                            replace=True)
        indices = torch.randint(0, len(x_cls), size=(len(x_cls), n_chunks), device=x_cls.device)

        for sample_idx, idx in enumerate(indices):
            # add offset
            idx += np.arange(0, chunks.shape[0], len(x_cls))

            # create new sample
            new_sample = chunks[idx]
            # 3D
            if x_cls.ndim == 3:
                new_sample = new_sample.permute(1, 0, 2).reshape(
                    1, x_cls.shape[1], x_cls.shape[2])
            else:  # 4D 
                new_sample = new_sample.permute(1, 2, 0, 3).reshape(
                    1, x_cls.shape[1], x_cls.shape[2], x_cls.shape[3])
            new_samples[current] = new_sample
            new_labels[current] = cls
            if subject_ids is not None:
                # Use the subject ID of the first chunk's sample
                new_subject_ids[current] = sub_cls[sample_idx % len(sub_cls)]
            current += 1

    combined_x = torch.cat((x, new_samples), dim=0)
    combined_y = torch.cat((y, new_labels), dim=0)
    if subject_ids is not None:
        combined_subject_ids = torch.cat((subject_ids, new_subject_ids), dim=0)

    # shuffle
    perm = torch.randperm(len(combined_x))
    combined_x = combined_x[perm]
    combined_y = combined_y[perm]

    if subject_ids is not None:
        combined_subject_ids = combined_subject_ids[perm]
        return combined_x, combined_y, combined_subject_ids

    return combined_x, combined_y

