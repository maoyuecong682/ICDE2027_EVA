# EVA Production Experiment Guide

## 1. Experiment Environment

| Item | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090 D |
| Python | 3.12.3, `/root/miniconda3/bin/python` |
| CUDA | 12.8 |
| PyTorch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 |
| NumPy / pandas | 2.3.2 / 3.1.0 |
| SciPy / scikit-learn | 1.18.1 / 1.9.1 |
| Pillow | 11.3.0 |
| transformers / clip | 4.57.6 / 1.0 |

## 2. Files and Directories

| Path | Description |
| --- | --- |
| `/root/chaichengliang/VLDB2027/EVA/EVA/methods/` | EVA methods and training/evaluation functions |
| `/root/chaichengliang/VLDB2027/EVA/EVA/mydatasets/` | Dataset loading and preprocessing |
| `/root/chaichengliang/VLDB2027/EVA/EVA/evaluation/eval_runner.py` | Method module loader |
| `/root/autodl-tmp/EVA/baseline/COPS-main/` | COPS baseline |
| `/root/autodl-tmp/EVA/baseline/GM-Matching-main/` | GM-Matching baseline |
| `/root/autodl-tmp/EVA/baseline/dual-pruning-main/exp_cifar/` | DUAL baseline |
| `/root/autodl-tmp/EVA/data/` | Datasets, splits, and preprocessing caches |
| `/root/autodl-tmp/eva_output/eva_adapter.py` | Shared dataset loading and model construction |
| `/root/autodl-tmp/eva_output/prepare_eva_embed.py` | GM-Matching feature generation |
| `/root/autodl-tmp/eva_output/clip_visual_cpu.py` | CLIP image encoder loader |
| `/root/autodl-tmp/eva_output/calibrate_adult_metric.py` | COPS selection threshold calibration |
| `/root/autodl-tmp/EVA/evaluation/eval_baselines.py` | Unified downstream training and evaluation |

Production experiment entry directory: `/root/autodl-tmp/EVA/evaluation/matrix/`.

| File or Directory | Description |
| --- | --- |
| `config.json` | Experiment parameters: datasets, retention fractions, random seeds, and training epochs |
| `run_all.sh` | Production experiment launcher |
| `runtime_check.py` | Environment, CUDA, and required input checks |
| `pipeline.py` | Orchestrates dataset loading, baseline preparation, selection, training, and evaluation |
| `sequence_worker.py` | Executes native baseline scripts |
| `status.py` | Reports experiment progress |
| `production_runs/<run_id>/` | Production experiment output directory |
| `manifest.json`, `effective_config.json` in the output directory | Experiment manifest, source hashes, and effective configuration |
| `prepare/` in the output directory | Baseline preparation artifacts |
| `indices.npy`, `selection.json` in each configuration directory | Selected subset indices and selection records |
| `kmeans.pth`, `result.json` in each configuration directory | Downstream model weights and evaluation results |
| `summary.csv`, `summary.json` in the output directory | Evaluation metric summaries |
| `timing.csv`, `timing_breakdown.csv` in the output directory | Total and per-stage execution times |
| `matrix_timing.json` in the output directory | Overall experiment execution time |

Existing production results: `/root/autodl-tmp/EVA/evaluation/matrix/production_runs/5978ee050174/`.

## 3. Running the Code

```bash
cd /root/autodl-tmp/EVA/evaluation/matrix

# Check the production experiment environment
/root/miniconda3/bin/python runtime_check.py --require-cuda

# Launch production experiments in the background using config.json
bash run_all.sh
```

Specify the GPU, Python executable, and output directory:

```bash
MATRIX_CUDA_DEVICE=0 \
MATRIX_PYTHON=/root/miniconda3/bin/python \
MATRIX_OUTPUT=/root/autodl-tmp/EVA/evaluation/matrix/production_runs \
bash /root/autodl-tmp/EVA/evaluation/matrix/run_all.sh
```

View progress and runtime logs:

```bash
cd /root/autodl-tmp/EVA/evaluation/matrix
/root/miniconda3/bin/python status.py
tail -f "$(cat production_runs/latest_log.txt)"
```
