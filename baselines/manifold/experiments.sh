#!/usr/bin/env bash
source "$(dirname "$0")/../../experiments/common.sh"

T="${THREADS:-4}"
ENV="OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T"
TP="$ENV $PY -m baselines.manifold.moabb_protocol"

LATTE_ROUTE="--subject_dropout_p 0.2 --mask_test_subject true"
cells() { for s in $(seq 1 "$1"); do echo "sub$s"; done; }

for c in $(cells 9); do
    echo "$TP --model tsmnet  --arm default --dataset BNCI2014001 --strategy inter-subject --cell $c"
    echo "$TP --model heegnet --arm default --dataset BNCI2014001 --strategy inter-subject --cell $c"
done

for c in $(cells 9); do
    echo "$TP --model gyroatt --dataset BNCI2014001 --strategy inter-subject --cell $c"
    echo "$TP --model gyroatt --input_align --dataset BNCI2014001 --strategy inter-subject --cell $c"
done

for arm in "" "--input_align --bn_refit"; do
    for c in $(cells 9); do
        echo "$TP --model latte --row base $arm $LATTE_ROUTE --dataset BNCI2014001 \
--strategy inter-subject --cell $c"
    done
done

for arm in "" "--input_align --bn_refit"; do
    for c in $(cells 12); do
        echo "$TP --model latte --row base $arm $LATTE_ROUTE --epochs 100 --label e100 \
--dataset BNCI2015001 --strategy inter-subject --cell $c"
    done
done
