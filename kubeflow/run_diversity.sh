#!/usr/bin/env bash
# Diversity of the DexGYS reproduction, following the DexGYS paper:
#   "We perform eight samples in the intention and diversity component under the same
#    input conditions, and each sample is individually sent to the quality component
#    for refinement." -> std of translation (cm), rotation and joints (Euler, deg).
#
#   per IDGC ckpt <I>:  test-set sampling with seed 3407+s, s = 0..N-1
#                       -> Experiments/eval/idgc_samples_<I>/s<s>   (shared by every QGC)
#   per variant <I>:<Q>: QGC ckpt <Q> (trained on IDGC <I>) refines each sample set,
#                       then the N refined sets are merged per input condition
#                       (obj_id, cate_id, action_id, guidance) and scored with
#                       scripts/diversity/calc_diversity.py
#   <I>, <Q> in {best, latest}, the same checkpoints as kubeflow/run_pipeline.sh.
#   seed 3407 (s=0) is the config seed, so sample 0 approximately reproduces
#   Experiments/results/idgc_<I>__qgc_<Q> (GPU nondeterminism -> not bit-exact).
#
# Output: Experiments/eval/idgc_<I>__qgc_<Q>/diversity/{all,dexter309}/diversity_with_std.json
# Usage:  VARIANTS="best:latest latest:best" bash kubeflow/run_diversity.sh
#         (default: all four variants; finished steps are skipped via .done markers)
set -euo pipefail
cd "$(dirname "$0")/.."

N=${N_SAMPLES:-8}
VARIANTS=${VARIANTS:-"best:best best:latest latest:best latest:latest"}
EV=./Experiments/eval
trap 'chown -R 1000:100 Experiments/eval 2>/dev/null || true' EXIT

# pick_ckpt <train save_root> <best|latest>  (same rule as run_pipeline.sh)
pick_ckpt() {
  python - "$1" "$2" <<'PY'
import glob, re, sys
root, which = sys.argv[1:]
if which == "latest":
    (c,) = glob.glob(f"{root}/epoch*_minus_loss_*_latest.pth")
    print(c)
else:
    cands = [(float(m.group(1)), p) for p in glob.glob(f"{root}/epoch*_minus_loss_*_.pth")
             if (m := re.search(r"_minus_loss_(-?[\d.]+)_\.pth$", p))]
    print(max(cands)[1])
PY
}

idgc_samples() {   # idgc_samples <I>: N seeded IDGC test-set samplings, once per IDGC ckpt
  local I=$1 ckpt; ckpt=$(pick_ckpt ./Experiments/idgc "$I")
  for s in $(seq 0 $((N - 1))); do
    local out="$EV/idgc_samples_$I/s$s"
    [ -f "$out.done" ] && continue
    echo "=================== IDGC[$I] sample $s / seed $((3407 + s)) ($(date '+%F %T')) ==================="
    rm -rf "$out"   # test.py would input()-prompt otherwise
    python ./test.py --train_cfg ./config/idgc.yaml --test_cfg ./config/infer_idgc_test.yaml \
      --override model.checkpoint_path "'$ckpt'" save_root "'$out'" seed "$((3407 + s))" < /dev/null
    touch "$out.done"
  done
}

for V in $VARIANTS; do
  I=${V%%:*}; Q=${V##*:}
  NAME="idgc_${I}__qgc_${Q}"; D="$EV/$NAME/diversity"
  mkdir -p "$D"
  idgc_samples "$I"
  QGC_CKPT=$(pick_ckpt "./Experiments/qgc_from_idgc_$I" "$Q")
  echo "##### $NAME   QGC: $QGC_CKPT"
  for s in $(seq 0 $((N - 1))); do
    [ -f "$D/qgc_s$s.done" ] && continue
    echo "=================== $NAME: QGC refine sample $s ($(date '+%F %T')) ==================="
    rm -rf "$D/qgc_s$s"
    IDGC_RES=$(ls "$EV"/idgc_samples_$I/s$s/test_results/*/matched_results.json)
    python ./test.py --train_cfg ./config/qgc.yaml --test_cfg ./config/infer_qgc_test.yaml \
      --override model.checkpoint_path "'$QGC_CKPT'" data.test.pose_path "'$IDGC_RES'" \
                 save_root "'$D/qgc_s$s'" seed "$((3407 + s))" < /dev/null
    touch "$D/qgc_s$s.done"
  done

  echo "=================== $NAME: merge + diversity ($(date '+%F %T')) ==================="
  python - "$D" "$N" "$NAME" <<'PY'
import json, glob, os, sys
import numpy as np
D, N, NAME = sys.argv[1], int(sys.argv[2]), sys.argv[3]
dexter = {x["obj_id"] for x in json.load(open(
    "/workspace/users/dmsgk724/dexter_temp/test_output/dexter_rot_bs16x4_ls1_final_test/predictions.json"))}
merged, n_per_run = {}, []
for s in range(N):
    (path,) = glob.glob(f"{D}/qgc_s{s}/test_results/*/raw_results.json")
    raw = json.load(open(path))
    n_per_run.append(sum(len(r["predictions"]) for r in raw))
    for r in raw:
        k = (r["obj_id"], r["cate_id"], r["action_id"], r["guidance"])
        merged.setdefault(k, dict(r, predictions=[]))["predictions"] += r["predictions"]
assert len(set(n_per_run)) == 1, n_per_run
# sanity: seed 3407 (s=0) vs the pipeline's result for this variant (close, not bit-exact)
(ref,) = glob.glob(f"Experiments/results/{NAME}/test_results/*/raw_results.json")
(s0,) = glob.glob(f"{D}/qgc_s0/test_results/*/raw_results.json")
a = np.array([p for r in json.load(open(ref)) for p in r["predictions"]]).reshape(-1, 28)
b = np.array([p for r in json.load(open(s0)) for p in r["predictions"]]).reshape(-1, 28)
print(f"seed-0 vs pipeline result: median trans diff {np.median(np.linalg.norm(a[:, :3] - b[:, :3], axis=1)) * 100:.2f} cm")
groups = list(merged.values())
print(f"{len(groups)} input conditions, predictions per condition: {sorted({len(g['predictions']) for g in groups})}")
for name, sel in [("all", groups), ("dexter309", [g for g in groups if g["obj_id"] in dexter])]:
    os.makedirs(f"{D}/{name}", exist_ok=True)
    json.dump(sel, open(f"{D}/{name}/raw_results_x{N}.json", "w"))
PY
  for SET in all dexter309; do
    rm -f "$D/$SET/diversity_with_std.json"
    python scripts/diversity/calc_diversity.py -r "$D/$SET/raw_results_x$N.json" < /dev/null
    python - "$D/$SET/diversity_with_std.json" "$NAME/$SET" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))["diversity"]["mean_object"]
print(f"[{sys.argv[2]}] δt {m['mean_std_trans']:.3f} cm | "
      f"δr {m['mean_std_rotation']:.3f} deg (wrap-aware {m['mean_real_std_rotation']:.3f}) | "
      f"δq {m['mean_std_joints']:.3f} deg (wrap-aware {m['mean_real_std_joint_angle']:.3f})"
      "   [mean over input conditions of per-condition std]")
PY
  done
done
echo "DIVERSITY DONE"
