# Reproduction: DexGYSGrasp on DexGYSNet (retrained)

DexGYSGrasp retrained from this code on the DexGYSNet dataset (torch 2.7.1 + cu128, B200; `Dockerfile.gys`) and
scored with the dexter_temp DexGYS benchmark. Reported setting: **IDGC latest (epoch 200) × QGC latest (epoch 100)**.
Con. and Q1 are raw × 10 (the paper's ×10⁻¹ unit); diversity = 8 seeded IDGC samplings per input condition, each
refined by QGC.

| Method | P-FID ↓ | CD ↓ | Con. ↓ | SR sim-only ↑ | SR E_pen² < 0.001 ↑ | SR max_pen < 5mm ↑ | Q1 ↑ | Pen. (cm) ↓ | δt (cm) | δr (deg) | δq (deg) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DexGYSGrasp (paper) | 5.60 | 1.20 | 0.36 | 63.31 * | | | 0.83 | 0.22 | 6.12 | 55.68 | 6.12 |
| **DexGYSGrasp (ours)** | 5.01 | 1.68 | 0.46 | 61.10 | 59.13 | 45.61 | 1.23 | 0.26 | 4.71 | 51.94 | 6.17 |

\* the paper reports one Success rate without saying which penetration filter it uses.

## Checkpoints

Private Hugging Face repo [`EunhaPark/dexgysgrasp`](https://huggingface.co/EunhaPark/dexgysgrasp) (log in with a
token that has access). Paths mirror `Experiments/`, so the commands below work unchanged after:

```bash
hf auth login   # once
hf download EunhaPark/dexgysgrasp --local-dir ./Experiments
```

| What | Checkpoint (`EunhaPark/dexgysgrasp`) | Config |
|---|---|---|
| IDGC (epoch 200, latest) | `idgc/epoch200_minus_loss_-2.6619_latest.pth` (1.9 GB) | `config/idgc.yaml` |
| QGC (epoch 100, latest; trained on the IDGC-latest outputs) | `qgc_from_idgc_latest/epoch100_minus_loss_-3.9982_latest.pth` (243 MB) | `config/qgc.yaml` |

## Evaluation commands

```bash
# 1) IDGC on the test set (seed 3407) -> coarse grasps
python ./test.py --train_cfg ./config/idgc.yaml --test_cfg ./config/infer_idgc_test.yaml \
  --override model.checkpoint_path "'./Experiments/idgc/epoch200_minus_loss_-2.6619_latest.pth'" \
             save_root "'./Experiments/my_eval/idgc'"
# 2) QGC refines them
python ./test.py --train_cfg ./config/qgc.yaml --test_cfg ./config/infer_qgc_test.yaml \
  --override model.checkpoint_path "'./Experiments/qgc_from_idgc_latest/epoch100_minus_loss_-3.9982_latest.pth'" \
             data.test.pose_path "'./Experiments/my_eval/idgc/test_results/epoch_200_test/matched_results.json'" \
             save_root "'./Experiments/my_eval/qgc'"
# 3) dexter benchmark format -> my_eval/pred/dexter309/predictions.json
python scripts/to_dexter_predictions.py \
  --matched ./Experiments/my_eval/qgc/test_results/epoch_100_test/matched_results.json --out-dir ./Experiments/my_eval/pred
```
`save_root` must not exist yet, otherwise `test.py` asks before deleting it.

```bash
# 4) P-FID, CD, Con., Q1, Pen. (dexter image)
source /python/.venv/bin/activate
cd /workspace/users/dmsgk724/dexter_temp && uv pip install -e .
PRED=<this repo>/Experiments/my_eval/pred/dexter309/predictions.json
DATA=/workspace/datasets/dexterp/dexgys_final
python -m benchmark.dexgys.chamfer --pred-path $PRED --data-path $DATA --gpu 0 --parallel 16
python -m benchmark.dexgys.q1      --pred-path $PRED --data-path $DATA --gpu 0 --parallel 16
python -m benchmark.dexgys.fid     --pred-path $PRED --data-path $DATA
# 5) Success rate (Isaac Gym, dexter_temp .venv-isaacgym)
python benchmark/dexgys/success_rate.py --pred-path $PRED --data-path $DATA --no_force --gpu 0 --parallel 8
# 6) Diversity: 8 IDGC samplings (seed 3407..3414), each refined by QGC
VARIANTS="latest:latest" bash kubeflow/run_diversity.sh
```
Prediction file of the reported run (cholab PVC):
`/workspace/users/dmsgk724/Grasp-as-You-Say/Experiments/eval/idgc_latest__qgc_latest/dexter309/predictions.json`.
Training follows "Usage" in `README.md`; `kubeflow/run_pipeline.sh` runs train → infer → QGC → test end to end.
