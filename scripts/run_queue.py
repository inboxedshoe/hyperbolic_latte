import argparse
import getpass
import os
import queue
import subprocess
import sys
import threading
import time


def foreign_gpus():
    """Indices of GPUs running processes that belong to someone else."""
    try:
        uuids = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                               capture_output=True, text=True, check=True).stdout
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                              capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return set()
    index = {u.split(",")[1].strip(): int(u.split(",")[0]) for u in uuids.strip().splitlines()}
    me, busy = getpass.getuser(), set()
    for line in apps.strip().splitlines():
        uuid, pid = [t.strip() for t in line.split(",")]
        owner = subprocess.run(["ps", "-o", "user=", "-p", pid], capture_output=True, text=True).stdout.strip()
        if owner and owner != me and uuid in index:
            busy.add(index[uuid])
    return busy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--per_gpu", type=int, default=1)
    ap.add_argument("--log_dir", default=None)
    ap.add_argument("--allow_shared", action="store_true")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    if not args.allow_shared:
        busy = foreign_gpus()
        for g in sorted(busy & set(gpus)):
            print(f"skipping GPU {g}: in use by another user")
        gpus = [g for g in gpus if g not in busy]
    if not gpus:
        sys.exit("no usable GPU")

    with open(args.jobs) as f:
        jobs = [l.strip() for l in f if l.strip() and not l.lstrip().startswith("#")]
    log_dir = args.log_dir or os.path.splitext(args.jobs)[0] + "_logs"
    os.makedirs(log_dir, exist_ok=True)

    q = queue.Queue()
    for i, job in enumerate(jobs):
        q.put((i, job))
    failed = []

    def worker(gpu):
        while True:
            try:
                i, job = q.get_nowait()
            except queue.Empty:
                return
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
            log = os.path.join(log_dir, f"{i:04d}.log")
            print(f"[gpu {gpu}] job {i}: {job}", flush=True)
            with open(log, "w") as f:
                f.write(job + "\n\n")
                f.flush()
                rc = subprocess.run(job + " --device cuda:0", shell=True, env=env,
                                    stdout=f, stderr=subprocess.STDOUT).returncode
            if rc:
                failed.append((i, job))
                print(f"[gpu {gpu}] job {i} FAILED (exit {rc}), see {log}", flush=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in gpus for _ in range(args.per_gpu)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"done: {len(jobs) - len(failed)}/{len(jobs)} succeeded in {(time.time() - t0) / 60:.1f} min")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
