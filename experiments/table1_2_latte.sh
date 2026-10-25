#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

n_subjects() { case "$1" in bcic2a|bcic2b) echo 9 ;; bcicha) echo 16 ;; esac; }
sc_seeds() { case "$1" in bcic2b) seeds 3330 3331 3332 3333 3334 ;;
                          *) seeds 3330 3331 3332 3333 3334 3335 3336 3337 3338 3339 ;; esac; }

for ds in bcic2a bcicha bcic2b; do
    echo "# LAtte (SC), $ds"
    for s in $(sc_seeds "$ds"); do
        echo "$PY scripts/train.py --cfg configs/sc/$ds.cfg --sub all $(data_flag "$ds") \
$VARIANT_FLAGS --seed $s --tag table1_2_sc --run_name sc_${ds}_s$s"
    done
done

for ds in bcic2a bcicha bcic2b; do
    echo "# LAtte (SS), per subject, $ds"
    for s in $(seeds 3330 3331 3332); do
        for sub in $(seq 1 "$(n_subjects "$ds")"); do
            echo "$PY scripts/train.py --cfg configs/ss/$ds.cfg --sub $sub $(data_flag "$ds") \
$VARIANT_FLAGS --seed $s --tag table1_2_ss --run_name ss_${ds}_sub${sub}_s$s"
        done
    done
done
