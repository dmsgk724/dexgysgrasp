#!/usr/bin/env bash
# Full README pipeline, run both for the best-val-loss checkpoint and for the
# last checkpoint at each stage (2 x 2 = 4 final results):
#
#   a) train IDGC                         -> $EXP/idgc/
#   b) infer IDGC on train + test, once per IDGC ckpt (best | latest)
#                                         -> $EXP/idgc_infer_<I>/test_results/epoch_<N>_{train,test}/
#   c) train QGC on each b) result        -> $EXP/qgc_from_idgc_<I>/
#   d) infer QGC on the matching IDGC test result, once per QGC ckpt (best | latest)
#                                         -> $EXP/results/idgc_<I>__qgc_<Q>/test_results/epoch_<M>_test/
#
#   <I>, <Q> in {best, latest}. best = highest minus_loss among the top-n saves,
#   latest = the *_latest.pth written at the end of training.
#   Every variant gets its own save_root, so best/latest never collide even when
#   they are the same epoch (test.py names its output dir only by epoch).
#
# Each finished step writes $EXP/pipeline/<step>.done, and a rerun skips steps
# that are already done. A step that crashed midway reruns from scratch.
# Delete a marker to force that step to rerun.
#
# Usage (from anywhere; cds to the repo root):
#   bash kubeflow/run_pipeline.sh            # full run, EXP=./Experiments, configs as-is
#   SMOKE=1 bash kubeflow/run_pipeline.sh    # every step on a tiny subset, 1 epoch each,
#                                            # EXP=./Experiments/pipeline_smoke (wiped first)
set -euo pipefail
cd "$(dirname "$0")/.."

SMOKE=${SMOKE:-0}
CFG=./config
EXP=./Experiments
if [ "$SMOKE" = 1 ]; then
  EXP=./Experiments/pipeline_smoke
  CFG=/tmp/gys_smoke_cfg
  rm -rf "$EXP" "$CFG"; mkdir -p "$CFG"
  # same configs, but tiny data subsets and 2 epochs (so best != latest is
  # possible and both code paths run); results under $EXP
  python - "$CFG" "$EXP" <<'PY'
import json, sys, yaml
cfg_dir, exp = sys.argv[1:]
for split, src, n in [("train", "data/dexgys/train.json", 256), ("test", "data/dexgys/test.json", 64)]:
    json.dump(json.load(open(src))[:n], open(f"{cfg_dir}/{split}.json", "w"))
for name in ["idgc", "qgc"]:
    c = yaml.safe_load(open(f"config/{name}.yaml"))
    c.update(epochs=2, validate_freq=1, print_freq=4, save_top_n=1, save_root=f"{exp}/{name}")
    c["scheduler"]["t_max"] = 2
    for s in ["train", "val"]:
        c["data"][s].update(batch_size=16, num_workers=2)
    if name == "idgc":
        c["data"]["train"]["pose_path"] = f"{cfg_dir}/train.json"
        c["data"]["val"]["pose_path"] = f"{cfg_dir}/test.json"
    yaml.safe_dump(c, open(f"{cfg_dir}/{name}.yaml", "w"), sort_keys=False)
for name in ["infer_idgc_train", "infer_idgc_test", "infer_qgc_test"]:
    c = yaml.safe_load(open(f"config/{name}.yaml"))
    c["data"]["test"].update(batch_size=16, num_workers=2)
    if name.startswith("infer_idgc"):
        c["data"]["test"]["pose_path"] = f"{cfg_dir}/{name.rsplit('_', 1)[1]}.json"
    yaml.safe_dump(c, open(f"{cfg_dir}/{name}.yaml", "w"), sort_keys=False)
print("smoke configs ->", cfg_dir)
PY
fi
echo "SMOKE=$SMOKE  CFG=$CFG  EXP=$EXP"

MARK=$EXP/pipeline
mkdir -p "$MARK"
# pods run as root; hand results back to the notebook user (jovyan 1000:100)
trap 'chown -R 1000:100 Experiments 2>/dev/null || true' EXIT

# pick_ckpt <train save_root> <best|latest>
pick_ckpt() {
  python - "$1" "$2" <<'PY'
import glob, re, sys
root, which = sys.argv[1:]
if which == "latest":
    c = glob.glob(f"{root}/epoch*_minus_loss_*_latest.pth")
    if len(c) != 1:
        sys.exit(f"expected exactly 1 *_latest.pth in {root}, found {c}")
    print(c[0])
else:
    cands = []
    for p in glob.glob(f"{root}/epoch*_minus_loss_*_.pth"):
        m = re.search(r"epoch(\d+)_minus_loss_(-?[\d.]+)_\.pth$", p)
        if m:
            cands.append((float(m.group(2)), p))
    if not cands:
        sys.exit(f"no top-n checkpoint in {root}")
    print(max(cands)[1])
PY
}
ckpt_epoch() { python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "$1"; }

step() { echo; echo "=================== $1 ($(date '+%F %T')) ==================="; }
done_() { touch "$MARK/$1.done"; }
is_done() { [ -f "$MARK/$1.done" ]; }

# train <name> <cfg> <save_root> [extra --override pairs...]
train() {
  local name=$1 cfg=$2 root=$3; shift 3
  is_done "$name" && return 0
  step "train $name -> $root"
  test -d "$root" && { echo "$root exists from an unfinished run -> moving aside"; mv "$root" "$root.incomplete.$(date +%s)"; }
  python train.py -t "$cfg" --override save_root "'$root'" "$@"
  done_ "$name"
}

# infer <name> <train cfg> <test cfg> <ckpt> <save_root> [extra --override pairs...]
infer() {
  local name=$1 tcfg=$2 icfg=$3 ckpt=$4 root=$5; shift 5
  is_done "$name" && return 0
  step "infer $name  ckpt=$ckpt -> $root"
  rm -rf "$root"   # test.py would input()-prompt if its output dir exists
  python ./test.py --train_cfg "$tcfg" --test_cfg "$icfg" \
    --override model.checkpoint_path "'$ckpt'" save_root "'$root'" "$@" < /dev/null
  done_ "$name"
}

VARIANTS="best latest"
# IDGC_VARIANTS picks which IDGC branches (b -> c -> d) this run executes.
# Running branches in parallel on 2 GPUs: once a) is done, submit a second job
# with IDGC_VARIANTS=latest. The branches are independent, and each one is
# claimed ($MARK/branch_<I>.claim = pod name) by the first job that reaches it,
# so the other job skips it. A crashed job leaves its claim behind: delete the
# .claim file before rerunning that branch from a new pod.
IDGC_VARIANTS=${IDGC_VARIANTS:-$VARIANTS}
ME=${HOSTNAME:-$(hostname)}
declare -A FINAL

# ---------------- a) IDGC train ----------------
if [ "$IDGC_VARIANTS" != "$VARIANTS" ] && ! is_done a_idgc_train; then
  echo "ERROR: IDGC_VARIANTS=$IDGC_VARIANTS needs a) IDGC training to be finished first"; exit 1
fi
train a_idgc_train "$CFG/idgc.yaml" "$EXP/idgc"

for I in $IDGC_VARIANTS; do
  CLAIM="$MARK/branch_$I.claim"
  # noclobber makes the claim atomic: only one job can create the file
  if ! ( set -o noclobber; echo "$ME" > "$CLAIM" ) 2>/dev/null && [ "$(cat "$CLAIM")" != "$ME" ]; then
    echo "branch idgc_$I is claimed by $(cat "$CLAIM") -> skipping it in this job"; continue
  fi
  IDGC_CKPT=$(pick_ckpt "$EXP/idgc" "$I"); IDGC_EP=$(ckpt_epoch "$IDGC_CKPT")
  echo "IDGC[$I]: $IDGC_CKPT (epoch $IDGC_EP)"

  # ---------------- b) IDGC infer (train + test) ----------------
  IROOT="$EXP/idgc_infer_$I"
  for SET in train test; do
    infer "b_idgc_${I}_infer_$SET" "$CFG/idgc.yaml" "$CFG/infer_idgc_$SET.yaml" "$IDGC_CKPT" "$IROOT/$SET"
  done
  TRAIN_RES="$IROOT/train/test_results/epoch_${IDGC_EP}_train/matched_results.json"
  TEST_RES="$IROOT/test/test_results/epoch_${IDGC_EP}_test/matched_results.json"
  test -f "$TRAIN_RES" && test -f "$TEST_RES"

  # ---------------- c) QGC train on this IDGC variant ----------------
  QROOT="$EXP/qgc_from_idgc_$I"
  train "c_qgc_from_idgc_${I}_train" "$CFG/qgc.yaml" "$QROOT" \
    data.train.pose_path "'$TRAIN_RES'" data.val.pose_path "'$TEST_RES'"

  # ---------------- d) QGC test, per QGC ckpt ----------------
  for Q in $VARIANTS; do
    QGC_CKPT=$(pick_ckpt "$QROOT" "$Q"); QGC_EP=$(ckpt_epoch "$QGC_CKPT")
    echo "QGC[idgc=$I, $Q]: $QGC_CKPT (epoch $QGC_EP)"
    RROOT="$EXP/results/idgc_${I}__qgc_${Q}"
    infer "d_idgc_${I}__qgc_${Q}_test" "$CFG/qgc.yaml" "$CFG/infer_qgc_test.yaml" "$QGC_CKPT" "$RROOT" \
      data.test.pose_path "'$TEST_RES'"
    FINAL[idgc_${I}__qgc_${Q}]="$RROOT/test_results/epoch_${QGC_EP}_test"
  done
done

step "PIPELINE DONE (${#FINAL[@]} results)"
for k in "${!FINAL[@]}"; do
  echo "$k: ${FINAL[$k]}"
  ls "${FINAL[$k]}"
done
