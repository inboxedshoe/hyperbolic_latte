#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"
RUNS="${1:-runs}"

for ckpt in $(find "$RUNS" -path "*sc_*" -name "best_val_suball.pt" | grep -v bcicha | sort); do
    run_dir="$(dirname "$(dirname "$ckpt")")"
    flags="$($PY - "$run_dir/params.json" <<'PYEOF'
import json, sys
p = json.load(open(sys.argv[1]))
keep = ["dataset", "linear_type", "batch_type", "pool_type", "windows", "lora_lr", "lr", "wd",
        "bs", "sub_rank", "decoder_rank", "proc_lora_lr", "preprocessor_adaptive", "interaug", "scheduler",
        "manifold_limits", "block_norm", "decoder_proj", "proto_init", "proto_std", "learn_decoder", "cutfill",
        "seed", "val_split", "data_path"]
print(" ".join(f"--{k} {p[k]}" for k in keep if p.get(k) is not None))
PYEOF
)"
    echo "$PY scripts/finetune_subject.py --checkpoint $ckpt $flags --iterations 50 \
--tag table1_2_ft --run_name ft_$(basename "$run_dir")"
done
