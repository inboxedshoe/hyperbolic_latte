import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THIRD_PARTY = os.environ.get("BASELINES_THIRD_PARTY",
                             os.path.join(ROOT, "baselines", "external"))


def third_party(name):
    """Put `baselines/external/<name>` on sys.path, or fail with how to get it."""
    path = os.path.join(THIRD_PARTY, name)
    if not os.path.isdir(path):
        raise ImportError(
            f"{name} not found at {path}. Expected it under "
            f"baselines/external/, or set "
            f"BASELINES_THIRD_PARTY to a directory that holds it.")
    if path not in sys.path:
        sys.path.insert(0, path)
    return path
