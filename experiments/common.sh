PY="${PY:-python}"
VARIANT="${VARIANT:-as_run}"
DATA="${DATA:-./data}"

case "$VARIANT" in
    as_run) VARIANT_FLAGS="" ;;
    paper)  VARIANT_FLAGS="--pool_type clean --block_norm block --decoder_proj frozen \
--proto_init wrapped_normal --learn_decoder false --cutfill true" ;;
    *) echo "unknown VARIANT=$VARIANT (as_run|paper)" >&2; exit 1 ;;
esac

data_flag() {
    case "$1" in
        bcicha) echo "--data_path $DATA/BCIcha/" ;;
        *)      echo "" ;;
    esac
}

seeds() {  # seeds <default list...>; SEEDS env wins
    if [ -n "${SEEDS:-}" ]; then echo "$SEEDS"; else echo "$@"; fi
}

