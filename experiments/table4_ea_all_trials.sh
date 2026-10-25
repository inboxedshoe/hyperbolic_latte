#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

for s in $(seeds 3330 3331 3332 3333 3334); do
    for ds in bcic2a bcic2b; do
        echo "$PY scripts/train_loso.py --cfg configs/loso/$ds.cfg --input_align_ea_all_trials true \
$VARIANT_FLAGS --seed $s --tag table4_ea --run_name ea_${ds}_s$s"
    done
done
