#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

for ds in bcic2a bcic2b bcicha; do
    ss="$(seeds 3330 3331 3332 3333 3334)"
    echo "# LAtte LOSO, $ds"
    for s in $ss; do
        echo "$PY scripts/train_loso.py --cfg configs/loso/$ds.cfg $(data_flag "$ds") \
$VARIANT_FLAGS --seed $s --tag table3_loso --run_name loso_${ds}_s$s"
    done
done
