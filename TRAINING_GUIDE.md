# RIGAT training and results generator, package version 2.0.2

For the organized results supplied with this package, read `README.txt` first. Existing T01-T12 tables trigger preservation mode: their values are copied into the report without recalculation. Added record sheets do not establish training completion. The training instructions below describe a new experiment with the required original datasets and models.

This package trains, calibrates, evaluates and measures computational cost from dataset text, then exports Excel results. The second program reads that workbook to produce 12 tables and Fig3–Fig12, retaining the output names used by generator version 1.6. Result values come from the executed pipeline. Version 1.6 scenario outputs are neither numerical inputs nor accuracy targets.

The two main commands are:

- `rigat_train.py`: dataset and pretrained models → checkpoints, predictions, logs, measurements and `RIGAT_Training_Results.xlsx`.
- `rigat_generate_actual.py`: training workbook → 12 CSV tables, a summary workbook and 600 DPI figures.

Keep both Python files in the same directory. The trainer imports metric and calibration functions from the actual-results generator. Once the workbook is available, figure generation does not require PyTorch, model weights or the original dataset.

Version 2.0.1 translates documentation and generated provenance notes into English. Training objectives, calibration calculations, evaluation metrics and report calculations are unchanged. See `ENGLISH_REVISION.md` for filename mappings and validation scope, and `OUTPUT_SCHEMA.md` for output/data requirements.

## Environment setup

Use your existing GPU environment. From Anaconda Prompt:

```powershell
conda activate gpu_env
cd "C:\path\RIGAT_Training_Actual"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m pip install -r requirements_training.txt
python -m pip install -r requirements_binoculars_gpu.txt
```

PyTorch must detect CUDA for GPU training. The Binoculars GPU dependencies include `bitsandbytes` and `accelerate`. Installing without `--upgrade` retains installed packages that already satisfy the version ranges. Versions actually used are recorded in the experiment configuration. Backend installation guidance is available in the [bitsandbytes documentation](https://huggingface.co/docs/bitsandbytes/installation).

The default dataset location is:

```text
C:\Users\LEGION\Train AI or Not\MAGE
```

All five CSVs must contain `text`, `label` and `src`: `train.csv`, `valid.csv`, `test.csv`, `test_ood_set_gpt.csv` and `test_ood_set_gpt_para.csv`. Label 0 means machine and label 1 means human. Source data is read without modification.

## Preflight checks

Check the dataset and RIGAT configuration:

```powershell
python rigat_train.py --preflight --output-dir "RIGAT_Actual_Run"
```

Check the full comparator plan:

```powershell
python rigat_train.py --full-suite --preflight --output-dir "RIGAT_Actual_Run"
```

Preflight reads metadata and checks validation splits, key dependencies, backbone configurations and devices. It does not train model weights. Configuration metadata may be fetched from Hugging Face if unavailable locally. Use `--local-files-only` to restrict access to the existing cache. The full suite needs configurations for DeBERTa, Longformer and both Falcon models.

For the dataset checked in the original package validation, the fixed split contained 319,071 training, 45,434 selection and 11,358 calibration documents. Validation uses label × domain stratification with split seed 2026, held constant across model seeds. These counts describe that dataset; they do not replace counts measured from different inputs.

## Staged execution and resuming

The full suite consists of nine Stage I configurations × five seeds, giving **45 independent training runs**. A5 uses the A4 checkpoint from the same seed. Binoculars is a single frozen-model evaluation without detector fine-tuning.

Start one A4 pipeline while retaining the full-suite plan:

```powershell
python rigat_train.py --full-suite --run-only A4 --seed-only 2026 --output-dir "RIGAT_Actual_Run"
```

Continue the plan with the same experiment arguments:

```powershell
python rigat_train.py --full-suite --resume --output-dir "RIGAT_Actual_Run"
```

Completed runs are skipped. Interrupted runs resume from checkpoints at gradient-accumulation boundaries. Each checkpoint stores weights, optimizer, scheduler, scaler, group weights, data-order position and RNG state. Saving defaults to every 500 updates; use `--checkpoint-every 100` if needed. After a run completes and results are saved, the temporary optimizer checkpoint is removed to save disk space; final `model.safetensors` weights remain. Set `--keep-optimizer-checkpoints` from the start to retain optimizer checkpoints.

Code hashes, configuration, package versions, dataset hashes and local model sources are locked. A code or configuration change requires a new experiment directory. This includes the English revision: use the original version 2.0.0 scripts to resume an existing version 2.0.0 experiment. Do not replace recorded hashes or edit an old configuration to bypass the guard. `--run-only` and `--seed-only` limit executed work and can be removed on resume. Include custom dataset/model-location arguments again when resuming.

Without `--full-suite`, defaults are A0, A1, A2, A2_DRO, A3 and A4 across five seeds: 30 independent training runs. Select methods explicitly if required:

```powershell
python rigat_train.py --methods A0 A4 Longformer DP-Net-adapted WaveDetect-adapted --seeds 2026 2027 2028 2029 2030 --output-dir "RIGAT_Selected"
```

`completion_summary.json` reports completed training, pending work, calibrator fits and Binoculars status. Only completed runs enter the workbook. Excel is rebuilt after each completed run so partial results can be inspected.

## Training and calibration defaults

| Component | Default |
| --- | --- |
| RIGAT, DP-adapted and Wave-adapted backbone | `microsoft/deberta-v3-base` |
| Longformer backbone | `allenai/longformer-base-4096` |
| Model seeds | 2026, 2027, 2028, 2029, 2030 |
| Stage I | 12 fixed epochs, final checkpoint, no early stopping |
| Batching | 1 document, accumulation 16, encoder chunk batch 1 |
| Optimizer | AdamW, encoder LR 2e-5, head LR 1e-4, decay 0.01 |
| Stabilization | 10% warmup, clipping 1.0, head dropout 0.1 |
| Precision | FP32 master parameters, CUDA FP16 autocast and GradScaler |
| Memory | Gradient checkpointing enabled |
| RIGAT objective | Class-balanced BCE + 0.25 local + 0.10 invariant + 0.50 robust; components depend on the ablation |
| GroupDRO | Domain × label groups, update rate 0.05 |
| A4 reference | Up to 12,000 training embeddings, class means, pooled within-class Ledoit–Wolf |
| Trust | NLL + 0.5 Brier + 1e-4 L2 on active parameters |
| Control calibrators | Temperature, Platt, beta, isotonic, entropy-only, distance-only |

Five completed A4 seeds give 35 main fits and 20 additional fits for K={1,2,4,6}; K=8 reuses the main fit. The Binoculars scale fit is a separate baseline fit. Counts follow actual executed work, including partial seed selections.

Distance uses representations and training statistics from each seed. The default is raw squared Mahalanobis distance, matching Eq. (11) as described in the original package. `--distance-normalization train-median` is available for a separate experiment; disclose its use because it changes feature scaling and the effect of regularization.

Special tokens, generator mapping, GroupDRO accumulation and comparator adaptations are detailed in `MANUSCRIPT_ALIGNMENT.md`.

## External datasets

The five MAGE CSVs do not include an external dataset. Supply an actual external CSV containing:

```text
Instance_ID,text,label,Domain,Generator
```

`Instance_ID` must uniquely identify a source example. Complete a copy of `external_protocol_template.json`, including at least dataset name, version, source, selection rule and selection seed. Record domains, generators and filters so the transfer protocol can be inspected.

```powershell
python rigat_train.py --full-suite --external-csv "D:\RAID\subset.csv" --external-protocol "D:\RAID\protocol.json" --output-dir "RIGAT_With_External"
```

Parameters and thresholds are frozen before external evaluation. Samples are not automatically selected to obtain a desired score. Without external data, T11 retains its headers and Fig11 is not generated. `T11_RAID_Validation` and `Fig11_RAID_Transfer` retain compatible names; these names do not verify dataset identity. Protocol metadata and the CSV hash identify the supplied source.

## Computational cost

Costs are measured during execution, with batch size 1, 50 warmup iterations and 1,000 timed iterations by default. Measurement uses a seed-determined document order. Close other GPU workloads during measurement so comparator conditions can be assessed.

| Quantity | Scope |
| --- | --- |
| `Training_Wall_Sec` | Token-cache retrieval, input preparation/transfer, forward, backward and optimizer; excludes initial tokenization, epoch validation, checkpoint I/O and export |
| `Additional_Stage_I_Wall_Sec` | Training time belonging to the configuration; A5 is 0 because it reuses A4 |
| `End_to_End_ms` | Raw-text tokenization, input preparation/transfer, model and probability mapping; A5 also includes distance and trust |
| `CUDA_Model_ms` | CUDA Events around forward/aggregation; Binoculars sums two LM intervals, including internal dispatch/offload |
| Peak allocated/reserved VRAM | PyTorch allocator during training/inference; differs from total device usage in NVIDIA-SMI |
| `Calibration_Fit_CPU_Sec` | Elapsed/wall-clock calibration optimizer time, rather than process CPU time |
| `Stage_II_A5_Compute_Wall_Sec` | Reference encoding, reference statistics, calibration-input encoding, calibration features and A5 fitting; excludes export/I/O and alternative calibrator fits |

Memory fields ending in `_GB` retain legacy table names but divide bytes by 1024³, giving GiB. CPU runs leave GPU measurements unavailable. Hardware profiles record the execution device rather than an assumed device. Raw timings retain `Timing_ID`; reusing K=8 timings does not represent a new measurement.

Binoculars uses Falcon-7B/Falcon-7B-Instruct with NF4 and CPU offload on CUDA. Default placement budgets per model are 2.5 GiB GPU and 10 GiB CPU, configurable with `--binoculars-gpu-budget-gb` and `--binoculars-cpu-budget-gb`. Check memory feasibility and backend support on the actual device. Both Falcon models in CPU FP32 do not fit a 32 GB RAM host. Original software validation used small causal models for the CPU path.

## Excel, tables and figures

After at least one run completes:

```powershell
python rigat_generate_actual.py --workbook "RIGAT_Actual_Run\RIGAT_Training_Results.xlsx" --output-dir "RIGAT_Actual_Report"
```

The generator writes 12 CSV tables and `RIGAT_Actual_Report.xlsx`. Fig3–Fig12 use 600 DPI; use `--dpi 150` for a preview. Fig11 requires external results, and other figures require their source sheets. The generator uses workbook content, including ROC/PR curves, reliability bins, risk-coverage curves and raw timings. `report_manifest.json` records the workbook hash and generator package versions.

T08 and Fig8 distinguish descriptive 512-word coverage from encoder-token coverage. Latency comes from measurements, replacing artificial budgets from an earlier development generator. No static cost values are inserted.

The training workbook contains per-seed results, calibration results, calibration logits, domain summaries, curves, raw timings, a run registry and a prediction manifest. Full per-document predictions and checkpoints are stored in sidecar files to avoid Excel's row limit. Data provenance and limitations appear in a single README-sheet row.

## Jupyter

Execute the files rather than pasting both programs into one cell:

```python
%run "C:/path/RIGAT_Training_Actual/rigat_train.py" --full-suite --preflight --output-dir "RIGAT_Actual_Run"
```

Kernel `-f ...json` arguments are handled. Remove `--preflight` to train; add `--resume` with the same experiment arguments to continue.

## Validation and interpretation limits

The original package reports software validation with small models, actual-text subsets and CPU execution. It reports a DeBERTa-v3-base CPU forward/backward API check. Full 12-epoch × five-seed training, 4,096-token Longformer GPU execution and both Falcon NF4/offload models on RTX 4070 were not executed when that package was prepared. Manuscript values must be replaced or confirmed from completed full experiments. `PACKAGE_VALIDATION.json` preserves those historical validation claims separately from checks performed for this English revision.

Related documentation: [PyTorch AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html), [Hugging Face model loading/checkpointing](https://huggingface.co/docs/transformers/main_classes/model), and [bitsandbytes quantization](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes).
