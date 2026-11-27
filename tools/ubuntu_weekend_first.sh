#!/usr/bin/env bash
# First research job, pinned across retries; successful results are never refitted.
set -euo pipefail
cd -- "$(dirname -- "$0")/.."
if (( $# != 0 )); then echo "Usage: $0" >&2; exit 64; fi
if [[ -e .runtime/WEEKEND_STOP ]]; then echo 'Weekend stopped locally' >&2; exit 69; fi
if [[ $(git branch --show-current) != ubuntu/weekend-20261127 ]]; then
    echo 'Expected the auxiliary-PC weekend branch' >&2; exit 69
fi
mkdir -p .runtime data/ubuntu3050-weekend-first-20261127
exec 9>.runtime/weekend-first.lock
flock --nonblock --conflict-exit-code 75 9 || exit 75
out=data/ubuntu3050-weekend-first-20261127
if [[ ! -f $out/choice ]]; then
    if timeout --kill-after=5s 75 tools/ubuntu_weekend_split.sh; then
        printf 'spatial\n' > "$out/choice"
    else
        status=$?
        if (( status == 65 || status == 75 )); then exit "$status"; fi
        printf 'position-hires\n' > "$out/choice"
    fi
fi
kind=$(cat "$out/choice")
report=$out/$kind.json
common=(--device cuda --cache-dir .runtime/spatial-qwen3vl-cache --seed 20261127 --steps 160)
case "$kind" in
    spatial)
        timeout --kill-after=5s 75 tools/ubuntu_weekend_split.sh
        command=(kwater_lab.spatial_probe "${common[@]}" --max-pixels 262144 --top-k 3
                 --manifest data/clean-split-v1/split_manifest_dev.json --report "$report")
        inputs=(kwater_lab/spatial_probe.py results/clean_split_v1_summary.json
                data/clean-split-v1/split_manifest_dev.json)
        valid='.folds == [1,2,3] and .seed == 20261127 and .fit_steps == 160 and .fit_device == "cuda"
               and (.per_fold | keys == ["1","2","3"])
               and all(.per_fold[]; .fit_photos > 0 and .validation_photos > 0
                   and all(.global_linear, .regional_mil_topk;
                       (.recall_at_1_numerator | type == "number")
                       and (.recall_at_1_denominator | type == "number")
                       and (.ap_denominator_codes | type == "number")))'
        ;;
    position-hires)
        command=(kwater_lab.ubuntu_position_control "${common[@]}" --max-pixels 393216
                 --fit-device cuda --out "$report")
        inputs=(kwater_lab/spatial_probe.py kwater_lab/ubuntu_position_control.py
                data/local-kit/sample-input/manifest.jsonl)
        valid='.photo_count == 10 and .tile_count == 160 and .learned_head_persisted == false
               and .fit_conditions == {seed:20261127,steps:160,learning_rate:0.03,weight_decay:0.02,device:"cuda"}
               and (.split.train_photo_hashes | length == 6) and (.split.heldout_photo_hashes | length == 4)
               and (.true_quadrant_heldout_accuracy | type == "number")
               and (.train_only_shuffled_quadrant_heldout_accuracy | type == "number")
               and (.accuracy_difference | type == "number")
               and (.comparison_gate == "ADVANTAGE" or .comparison_gate == "NO_ADVANTAGE")'
        ;;
    *) echo 'Unknown persisted first job' >&2; exit 65 ;;
esac
inputs+=(kwater_lab/vlm_11code.py data/baseline/base_model/config.json
         data/baseline/base_model/preprocessor_config.json data/baseline/base_model/model.safetensors.index.json)
signature=$({ printf '%s\0' "${command[@]}"; sha256sum "${inputs[@]}"; } | sha256sum | cut -d ' ' -f 1)
if [[ -e $out/completed.sha256 ]]; then
    if [[ $(cat "$out/command.sha256") == "$signature" ]] &&
       [[ $(cat "$out/completed.sha256") == "$(sha256sum "$report")" ]] &&
       jq -e "$valid" "$report" >/dev/null; then
        echo "First research job already complete: $report"
        exit 0
    fi
    echo 'Completed result or its execution inputs changed; review before rerunning' >&2
    exit 65
fi
printf '%s\n' "$signature" > "$out/command.sha256"
# Remove only this job's incomplete report; content-addressed visual grids survive retries.
rm -f -- "$report"
tools/ubuntu_weekend_gpu.sh run "${command[@]}"
jq -e "$valid and (.cache.encoder_sha256 | test(\"^[0-9a-f]{64}$\"))" "$report" >/dev/null
current=$({ printf '%s\0' "${command[@]}"; sha256sum "${inputs[@]}"; } | sha256sum | cut -d ' ' -f 1)
if [[ $current != "$signature" ]]; then echo 'Execution inputs changed during the run' >&2; exit 65; fi
sha256sum "$report" > "$out/completed.sha256.tmp"
mv -T "$out/completed.sha256.tmp" "$out/completed.sha256"
echo "First research job complete: $report"
