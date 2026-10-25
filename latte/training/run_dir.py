import os
import json
import shutil
from datetime import datetime


def make_run_dir(base_dir, run_name, label=None, params=None, save_models_dir=None):
    """Create a timestamped run folder with a checkpoints subfolder.

    Structure:
        {base_dir}/{timestamp}_{run_name}[_{label}]/
            params.json
            models/          (optional snapshot of the models folder)
            results.csv      (written later by save_output)
            checkpoints/     (model checkpoints)

    Returns (run_dir, checkpoints_dir).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{timestamp}_{run_name}"
    if label:
        folder_name += f"_{label}"

    run_dir = os.path.join(base_dir, folder_name)
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    if params is not None:
        with open(os.path.join(run_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=2, default=str)

    if save_models_dir and os.path.isdir(save_models_dir):
        shutil.copytree(save_models_dir, os.path.join(run_dir, "models"),
                        dirs_exist_ok=True)

    return run_dir, checkpoints_dir
