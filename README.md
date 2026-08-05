# State AI

## Setup

```bash
pip install -e .
```

## Go

```bash
# 1. pick your folders
state-ai taxonomy seed --out my_taxonomy.json

# 2. dry-run 
state-ai organize-open ~/Desktop/MessyCopy \
  --taxonomy my_taxonomy.json \
  --checkpoint models/model.pt
```

Copy the folder first. Add `--apply --calibration models/calibration.json` only when the plan looks right.

## Extra

```bash
state-ai scan ~/Desktop/MessyCopy
state-ai taxonomy discover ~/Desktop/MessyCopy --taxonomy my_taxonomy.json
```

Training scripts: `scripts/`. Older baseline: `Baselines/old/v1/`.

## URTC multi-backend eval results

Synced evaluation package lives at:

- [`artifacts/urtc_eval_package/README_RESULTS.md`](artifacts/urtc_eval_package/README_RESULTS.md)
- [`artifacts/urtc_eval_package/ready_runs/TEAMMATE_RESULTS_PACK.md`](artifacts/urtc_eval_package/ready_runs/TEAMMATE_RESULTS_PACK.md)
- Tables: `artifacts/urtc_eval_package/ready_runs/rollup/PAPER_TABLES.md`
