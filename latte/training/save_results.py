import os
import csv
import numpy as np


def save_output(run_dir, results, **kwargs):
    """Write results.csv into an already-created run_dir."""
    output_file = os.path.join(run_dir, "results.csv")

    outputs = ['train_losses', 'train_accs', 'val_losses', 'val_accs', 'test_losses', 'test_accs']
    flat_results = [[float(x) for x in np.ravel(arr)] for arr in results]
    rows = list(zip(*flat_results))

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(outputs)
        writer.writerows(rows)
