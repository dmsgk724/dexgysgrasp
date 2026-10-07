#!/usr/bin/env bash
# Computational metrics of one DexGYS reproduction variant, scored with dexter_temp's
# DexGYS benchmark exactly as dexter is scored. Runs inside the dexter image
# (source /python/.venv/bin/activate first):
#   chamfer -> hand_chamfer (CD), cmap (contact map)
#   q1      -> valid_q1 (Q1, zeroed if pen > 5mm), mean_pen (max pen depth, m) + q1.csv
#   fid     -> fid (P-FID; point-e pointnet from dexter_temp/point_e_model_cache)
# Inputs (dexter predictions.json schema); results are written next to each file:
#   Experiments/eval/<VARIANT>/{dexter309,all}/predictions.json
# Usage: VARIANT=idgc_best__qgc_best bash run_dexter_metrics.sh
set -eo pipefail
: "${VARIANT:?set VARIANT, e.g. idgc_best__qgc_latest}"
E=/workspace/users/dmsgk724/Grasp-as-You-Say/Experiments/eval/$VARIANT
DATA=/workspace/datasets/dexterp/dexgys_final
trap 'chown -R 1000:100 $E 2>/dev/null || true' EXIT
cd /workspace/users/dmsgk724/dexter_temp
git config --global --add safe.directory '*'
git log -1 --oneline
nvidia-smi --query-gpu=name --format=csv,noheader
uv pip install -e .

for SET in dexter309 all; do
  PRED=$E/$SET/predictions.json
  echo "=================== $VARIANT / $SET ($(date '+%F %T')) ==================="
  rm -f "$E/$SET/metrics.json"
  python -m benchmark.dexgys.chamfer --pred-path $PRED --data-path $DATA --gpu 0 --parallel 16
  python -m benchmark.dexgys.q1      --pred-path $PRED --data-path $DATA --gpu 0 --parallel 16
  python -c "import csv,sys; p=[float(r['pen']) for r in csv.DictReader(open(sys.argv[1]))]; print(f'Q1 max<5mm {sum(x<0.005 for x in p)/len(p)*100:.2f}% (n={len(p)})')" $E/$SET/q1.csv
  python -m benchmark.dexgys.fid     --pred-path $PRED --data-path $DATA
  echo "--- $VARIANT / $SET metrics.json"; cat $E/$SET/metrics.json
done
echo "EVAL DONE"
