# Output schema and reconstruction requirements

This document maps the supplied Python code to its outputs. It describes what can be recomputed when training files arrive; it does not assert that any training run has completed. No training results, checkpoints or empty completion markers are included in this source package.

## Canonical directories and filenames

```text
RIGAT_Training_Actual/
├── rigat_train.py
├── rigat_generate_actual.py
├── requirements_training.txt
├── requirements_binoculars_gpu.txt
├── TRAINING_GUIDE.md
├── MANUSCRIPT_ALIGNMENT.md
├── external_protocol_template.json
├── PACKAGE_VALIDATION.json
├── ENGLISH_REVISION.md
├── OUTPUT_SCHEMA.md
├── RIGAT_Actual_Run/                 # Created by actual training
│   ├── RIGAT_Training_Results.xlsx
│   ├── README_results.md
│   ├── experiment_config.json
│   ├── preflight.json
│   ├── completion_summary.json
│   ├── input_manifest.csv
│   ├── source_mapping.csv
│   ├── train_ids.csv
│   ├── valid_select_ids.csv
│   ├── valid_calib_ids.csv
│   ├── external_protocol.json        # Only with external input
│   ├── source/
│   ├── tokenizer/
│   ├── tokenizer_longformer/         # Only when Longformer is planned
│   ├── token_cache/
│   └── runs/
│       ├── 2026_A0/                  # Same pattern for each planned seed/method
│       ├── 2026_A4/
│       │   ├── model.safetensors
│       │   ├── encoder_config/
│       │   ├── pretrained_loading_info.json
│       │   ├── reference.npz
│       │   ├── reference_ids.csv
│       │   ├── calibrators.json
│       │   ├── coverage_calibrators.json
│       │   ├── Standard.csv.gz
│       │   ├── Standard_A5.csv.gz
│       │   ├── OOD_GPT.csv.gz
│       │   ├── OOD_GPT_A5.csv.gz
│       │   ├── Paraphrase.csv.gz
│       │   ├── Paraphrase_A5.csv.gz
│       │   ├── coverage_raw_K1.csv.gz
│       │   ├── coverage_calibration_K1.csv.gz
│       │   ├── records_Classification.csv.gz
│       │   ├── records_Calibration.csv.gz
│       │   ├── records_Cost.csv.gz
│       │   ├── records_<sheet>.csv.gz
│       │   └── run_complete.json
│       ├── 2026_Longformer/
│       ├── 2026_DP-Net-adapted/
│       ├── 2026_WaveDetect-adapted/
│       └── frozen_Binoculars/
└── RIGAT_Actual_Report/              # Created by the report generator
    ├── RIGAT_Actual_Report.xlsx
    ├── report_manifest.json
    ├── T01_Comparator_Overview.csv
    ├── ...                          # T02–T12 listed below
    ├── Fig3_Ablation_MacroF1.png
    ├── ...                          # Fig4–Fig11 listed below
    └── Fig12_Model_Size_vs_F1.png
```

The directory is selected by `--output-dir`. `RIGAT_Actual_Run` is a naming convention, not a requirement to rename existing experiments. An A4 directory owns the A5 outputs; there is no separately trained A5 directory. Other methods do not have A4-only reference/calibration files. Coverage sidecars are omitted with `--skip-coverage`; K-specific calibration sidecars exist for restricted K values, while the maximum K reuses the main calibration data. `resume.pt` exists during training or when explicitly retained. External predictions are conditional on external input and the method's external-evaluation protocol.

## Training workbook sheets

The full configured pipeline can produce 20 sheets including README. Sheets assembled from run records are present only when those records exist. Column lists below identify core fields, not every optional diagnostic/hardware field.

| Sheet | Core fields or contents | Source / reconstruction requirement |
| --- | --- | --- |
| `README` | A1=`Data provenance`, B1=provenance paragraph; no header row | Configuration, completion counts and known measurement scope |
| `Classification` | Seed, Method, Condition, N, TP_Machine, FN_Machine, FP_Human, TN_Human; classification and calibration metrics | Per-document labels and logits; confusion counts alone recover only threshold-dependent metrics |
| `Calibration` | Seed, Method, Condition, Calibrator, N; same metric fields | Evaluation labels and transformed logits for each calibrator |
| `Bins` | Seed, Condition, Method, Calibrator, Bin, Bin_Low, Bin_High, N_bin, Mean_Confidence, Empirical_Accuracy | Per-document labels and calibrated confidence, or saved original bins |
| `Risk` | Seed, Condition, Method, Calibrator, Coverage, Selective_Risk | Labels and ranked absolute logits, or original risk curves |
| `Domain` | Seed, Method, Condition, Domain, N; counts and metrics | Per-document domain, label and logit, or original domain records |
| `Coverage` | Seed, Protocol, Kmax, N; token/word coverage and prediction metrics | Restricted-K predictions and coverage statistics; original word counts for word coverage |
| `Calibration_Diagnosis` | Seed, Calibrator, N_Validation, before/after metrics, parameters, objective and optimizer diagnostics | Saved fit metadata and valid_calib predictions/features; refitting is a new fit, not the historical optimizer run |
| `Coverage_Fits` | Kmax plus calibration diagnosis fields | Saved restricted-K calibration fits |
| `Validation_Logits` | Seed, Instance_ID, label, Domain, Raw_Logit, Entropy, Distance, coverage fields; one transformed-logit column per calibrator | Actual A4 valid_calib inference and fitted transformations |
| `Timing` | Seed, Method, Kmax, Timing_ID, Timing_Run, Instance_ID, End_to_End_ms, CUDA_Model_ms, Selected_Token_Chunks, Human_Probability | Actual individual timing measurements |
| `Coverage_Timing` | Timing fields plus Protocol and Kmax | Actual restricted-K timings; maximum-K measurements may be reused with the same Timing_ID |
| `ROC_PR` | Seed, Method, Condition, Grid, ROC_TPR, PR_Precision | Actual labels and logits, or saved 201-point interpolated curves |
| `Cost` | Seed, Method, Independent_Training_Run_Count, Training_Wall_Sec, Additional_Stage_I_Wall_Sec, Training_Cost_Parent, Total_Params_M; memory, hardware and calibration fields | Recorded resource measurements and actual model/configuration metadata |
| `History` | Seed, Method, Epoch, Accumulation_Windows, Optimizer_Updates, AMP_Skipped_Updates, Training_Elapsed_Sec, Selection_Macro_F1, Selection_NLL, Gradient_Norm_Last | Actual epoch logs or saved training state; final scores cannot recover an epoch history |
| `Predictions` | Seed, Method, File, SHA256, Instances | Existing prediction sidecars; hashes must be recomputed from the actual delivered files |
| `Baseline_Fits` | Method, Fit_CPU_Sec, Objective, Success, Score_Threshold, Positive_Scale, N_Calibration | Actual Binoculars mapping and fit metadata |
| `Run_Registry` | identity, Seed, Method, Independent_Training_Run_Count, Epochs, Checkpoint, Checkpoint_SHA256 | Actual run_complete.json metadata and existing checkpoints |
| `Dataset_Composition` | Split, Instances, Human, Machine, Domains | Actual partition data or original configuration partition_composition |
| `Configuration` | Component, Value; nested settings flattened into paths | Original experiment_config.json; do not silently substitute current script defaults |

All sheets other than README have a header row. Excel allows at most 1,048,575 data rows plus the header. Full per-document predictions belong in compressed sidecars when necessary. A summary metric row cannot be expanded into authentic per-document predictions.

## Calculation rules used by the scripts

- Label 0 is machine; label 1 is human. `Logit` is the human-oriented logit. Human probability is sigmoid(Logit), clipped to [1e-12, 1−1e-12]. A nonnegative logit predicts human.
- TP_Machine = actual machine predicted machine; FN_Machine = actual machine predicted human; FP_Human = actual human predicted machine; TN_Human = actual human predicted human. The four counts sum to N.
- Accuracy = (TP_Machine + TN_Human) / N. Macro-F1 is the mean of the two class F1 scores, with zero_division=0. Human_FPR = FP_Human / max(1, FP_Human + TN_Human). MCC follows `sklearn.metrics.matthews_corrcoef`.
- AUROC and average precision/AUPRC use machine-positive labels `1−label` and machine-oriented scores `−Logit`. A confusion matrix does not contain the score ranking required for these metrics.
- NLL = mean(logaddexp(0, Logit) − label × Logit). Brier = mean((clipped sigmoid(Logit) − label)²).
- ECE uses 15 equally spaced confidence bins over [0.5, 1.0], with confidence=max(p,1−p). Sum `(N_bin/N) × abs(Empirical_Accuracy−Mean_Confidence)` over nonempty bins.
- AURC ranks by descending abs(Logit), assigns expected errors within exact confidence ties and averages cumulative selective risks over all retained document counts. The exported Risk sheet samples 100 coverage levels from 0.01 through 1.0; averaging only these sampled points does not exactly reconstruct the full AURC.
- Raw entropy is binary entropy normalized by ln(2), computed from raw logits with the script's probability clipping. Reference distance is the minimum squared Mahalanobis distance to the two training class means divided by the recorded scale.
- A5 transforms raw A4 logits by max(sigmoid(Bias−Alpha×Entropy−Beta×Distance),1e-12)/T. This factor is positive. At the fixed 0.5 probability threshold, A4 and A5 therefore have identical class decisions, confusion matrices, accuracy, MCC and Macro-F1. Probability calibration and score rankings may change.
- Aggregates use the mean and sample SD across available seed rows; a single seed has unavailable SD. Binoculars is evaluated once: its summary n_runs is 1, with unavailable SD, even though paired-statistics rows repeat the fixed result across the seed schedule.
- Paired comparisons align Seed within Condition. T09 uses paired differences, nominal 95% t intervals and Holm correction within Family × Condition × Metric. Per-seed observations are required; mean/SD alone do not recover the paired covariance.
- Latency p50 is the raw sample median and p95 is the pandas 0.95 quantile per seed, before averaging seed-level summaries. A mean latency alone cannot recover those quantiles. `_GB` memory fields use 1024³ bytes.

## Report table mapping

The report workbook has README plus the following 12 sheets; each table also has an identically named CSV.

| Table / sheet | Source |
| --- | --- |
| `T01_Comparator_Overview` | Methods present in Standard Classification |
| `T02_Dataset_Composition` | Dataset_Composition |
| `T03_Experimental_Configuration` | Configuration |
| `T04_Ablation_and_Baselines` | Available Standard methods and recorded configuration |
| `T05_Standard_Test` | Standard Classification, seed mean/sample SD |
| `T06_Robustness_Shifts` | Classification; per-seed Standard−OOD and Standard−Paraphrase drops |
| `T07_Calibration_Trust` | Calibration; includes A5−Distance-Only deltas |
| `T08_Coverage_Efficiency` | Coverage and optional Coverage_Timing |
| `T09_Statistical_Comparisons` | Classification and Calibration, aligned seed pairs |
| `T10_Calibration_Diagnosis` | Calibration_Diagnosis |
| `T11_RAID_Validation` | External Classification only; dataset identity comes from protocol metadata |
| `T12_Computational_Cost` | Cost and optional raw Timing |

## Figure mapping

| PNG filename | Required information |
| --- | --- |
| `Fig3_Ablation_MacroF1.png` | Standard ablation Classification |
| `Fig4_Confusion_Matrices.png` | Confusion counts for the chosen method/seed across conditions |
| `Fig5_ROC_PR_Curves.png` | ROC_PR curves for the selected seed |
| `Fig6_Robustness_and_Domain.png` | Classification across conditions and Domain records |
| `Fig7_ECE_AURC_Bar.png` | Calibration ECE/AURC by condition and calibrator |
| `Fig8_Latency_Coverage.png` | Coverage plus required latency, token-coverage and word-coverage columns |
| `Fig9_Reliability_Diagrams.png` | Actual Bins records |
| `Fig10_Risk_Coverage_Curves.png` | Actual Risk records |
| `Fig11_RAID_Transfer.png` | Actual External Classification |
| `Fig12_Model_Size_vs_F1.png` | Standard Classification, parameter counts and measured Stage I time |

## Handling the training results supplied later

1. Preserve original values/files and identify the experiment version, seeds, methods, conditions, label orientation and units.
2. Map incoming file, sheet and column names to this schema using their content. Keep a mapping of original names to delivered names.
3. Split available per-seed/per-method records into the corresponding run folders. Preserve prediction-level IDs and alignment.
4. Recompute missing metrics only where required inputs exist, using the script functions. Record the source and derivation for reconstructed fields/sheets.
5. Recompute hashes after any authorized filename/path/content changes. Preserve original hashes and provenance in the mapping. A reconstructed report archive must not masquerade as an original resumable training run.
6. Check duplicate keys, confusion totals, A4/A5 decision agreement, domain totals and risk-coverage endpoints before building reports.
7. Report specific missing source files for anything that cannot be recovered. Model weights, tokenizer assets, token caches, reference embeddings/matrices, measured timings, optimizer history and original dataset hashes cannot be invented from summary Excel scores. Mark unavailable information explicitly instead of substituting zero or declaring an incomplete run complete.
