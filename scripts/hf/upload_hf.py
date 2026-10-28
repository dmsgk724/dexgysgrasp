"""
Publish the reported DexGYSGrasp checkpoints (IDGC latest x QGC latest) to the private Hugging Face model repo
EunhaPark/dexgysgrasp. Paths in the repo mirror Experiments/, so `hf download EunhaPark/dexgysgrasp --local-dir
./Experiments` restores the layout the README commands expect. Needs HF_TOKEN (Kubeflow: secret hf-secret-eunha).

    python scripts/hf/upload_hf.py --expect-user EunhaPark
"""
import argparse
import os
import sys

from huggingface_hub import HfApi

FILES = [
    "idgc/epoch200_minus_loss_-2.6619_latest.pth",                   # IDGC, epoch 200 (latest)
    "qgc_from_idgc_latest/epoch100_minus_loss_-3.9982_latest.pth",   # QGC trained on IDGC-latest outputs, epoch 100
]

CARD = """---
license: other
tags: [dexterous-grasping, dexgys, dexgysgrasp]
---
# DexGYSGrasp on DexGYSNet (reproduction)

Retrained DexGYSGrasp (*Grasp as You Say*, NeurIPS 2024, arXiv 2405.19291) on the DexGYSNet dataset with the
official code (Grasp-as-You-Say). Reported setting: IDGC latest x QGC latest.
Results and evaluation commands: Grasp-as-You-Say `README.md`, "Reproduction" section.

| File | What |
|---|---|
| `idgc/epoch200_minus_loss_-2.6619_latest.pth` | Intention and Diversity Grasp Component, epoch 200. Config `config/idgc.yaml` |
| `qgc_from_idgc_latest/epoch100_minus_loss_-3.9982_latest.pth` | Quality Grasp Component trained on the IDGC-latest outputs, epoch 100. Config `config/qgc.yaml` |

```bash
hf download EunhaPark/dexgysgrasp --local-dir ./Experiments   # inside Grasp-as-You-Say
```
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="EunhaPark/dexgysgrasp")
    parser.add_argument("--expect-user", default="EunhaPark")
    parser.add_argument("--root", default="./Experiments")
    args = parser.parse_args()

    api = HfApi(token=os.environ["HF_TOKEN"])
    user = api.whoami()["name"]
    if user != args.expect_user:
        sys.exit(f"token belongs to {user!r}, not {args.expect_user!r}: nothing uploaded")
    for f in FILES:
        assert os.path.isfile(os.path.join(args.root, f)), f

    api.create_repo(args.repo, repo_type="model", private=True, exist_ok=True)
    assert api.repo_info(args.repo).private, "repo is not private: refusing to upload"
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=args.repo,
                    commit_message="Model card")
    for f in FILES:
        if f in api.list_repo_files(args.repo):
            continue
        print("uploading", f, flush=True)
        api.upload_file(path_or_fileobj=os.path.join(args.root, f), path_in_repo=f, repo_id=args.repo,
                        commit_message=f"Add {f}")
    print(args.repo, "private:", api.repo_info(args.repo).private, "files:", api.list_repo_files(args.repo))


if __name__ == "__main__":
    main()
