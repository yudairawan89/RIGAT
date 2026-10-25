# Implementation alignment with the revised draft

The original package identifies `RIGAT_Revised_Manuscript - Copy.docx`, especially Sections 3.1–3.9, as its primary reference. That manuscript is not included in this ZIP; the statements below translate the supplied documentation and have not been checked against the manuscript itself. This document describes implementation choices to include in reproducibility materials and, where relevant, clarify in the manuscript. Draft values are not output targets.

## RIGAT and data partitioning

| Draft section | Implementation |
| --- | --- |
| 3.1 | Label 0 machine / 1 human; 80/20 validation split stratified by label × domain, seed 2026 |
| 3.2 | Shared DeBERTa-v3-base; uniform chunk selection including first/last for K≥2; positional embeddings follow selected-chunk order |
| 3.3 | Local MLP, binary entropy / ln(2), attention score + softplus(gamma) × log(reliability + 1e-6), LayerNorm |
| 3.4 | GRL, domain head for all documents, generator head only for machine documents; class-balanced BCE and domain × label GroupDRO |
| 3.5 | Training-derived class means and shared Ledoit–Wolf precision; per-seed distance; trust and controls use the same inputs |
| 3.6 | 12 fixed epochs; FP32 master parameters, CUDA FP16 autocast, accumulation 16, final checkpoint |
| 3.8 | Macro-F1, accuracy, MCC, machine-positive AUROC/AP, Human FPR; NLL/Brier/ECE/AURC; paired t-tests and Holm |
| 3.9 | Separate word/token coverage; two K-calibration protocols; measured costs with stated scope |

For an encoder input limit of 512, the CLS/SEP special tokens are included in the limit. With nominal overlap 64, the start stride remains 448, text payload is 510 tokens and content overlap is 62 tokens. Clarify this convention so “512 tokens / overlap 64” does not equate text payload with total encoder input length. Longformer uses a total input limit of 4,096 tokens with global attention on CLS.

The default generator mapping has six categories: OpenAI, LLaMA, OPT, FLAN_T5, GPT_Open and OtherMachine. T0, BLOOM, GLM and unmatched names map to OtherMachine. The complete source → category mapping is saved; categories are not prediction features. This explicit grouping was chosen because the draft specifies the count without detailing the complete mapping.

GroupDRO operates on documents within one accumulation window. Each document contributes q_group / number of documents in its group, so gradient accumulation corresponds to the sum of mean risks over observed groups. Global group weights are not renormalized within a single micro-batch. Window risk updates q for the next window. This retains q's effect when a micro-batch contains one document.

Reference precision is computed by Ledoit–Wolf estimation on embedding residuals relative to class means. Default distance is the minimum squared Mahalanobis distance without additional normalization. Statistics use only that seed's training reference and are not updated on calibration or test data. Reference identities and matrices are saved per seed.

Trust uses bounds log(T) [-3,3], bias [-8,12] and active coefficients [0,8], with three starts. L2 applies to log(T), bias and active coefficients. Temperature/Platt/beta use NLL; isotonic uses MSE/PAVA. All optimizer statuses are saved, including nonconvergence. A5 is not required to outperform controls, entropy is not required to be nonzero, and seeds are not required to produce different results.

A0 is a prefix classifier. A1 adds positional chunk embeddings, hierarchical mean aggregation and LayerNorm. A2 adds local weak supervision and reliability aggregation. A2_DRO adds GroupDRO without an adversary. A3 uses an adversary without GroupDRO. A4 combines both. A5 adds only post-hoc trust to A4, with a fixed probability threshold of 0.5.

## DP-Net-adapted

Original DP-Net learns Gaussian perturbations under reinforcement-learning control. The authors' implementation uses DDPG, providing the dynamic-perturbation reference. See the [DP-Net paper](https://arxiv.org/html/2504.21019v1) and [authors' repository](https://github.com/CAU-ISS-Lab/AIGT-Detection-Evade-Detection/tree/main/DP-Net).

This package adapts DeBERTa-v3-base fine-tuned on a 512-token prefix. A small policy observes detached CLS embeddings and selects a Gaussian scale from {0, 0.01, 0.03, 0.06, 0.10}, relative to embedding RMS. The feature extractor receives clean and perturbed embeddings. The objective is 0.5 clean BCE + 0.5 perturbed BCE + 0.01 alignment MSE. The policy uses REINFORCE with the negative objective as reward, EMA baseline 0.95, policy weight 0.01 and entropy bonus 0.01. Noise and policy are disabled during inference.

This is a discrete adaptation with a stated encoder and MAGE budget. The manuscript should disclose REINFORCE and the perturbation location; the implementation does not support a claim of replicating DDPG or the full original DP-Net configuration. Adaptation parameters are recorded and can be set before the experiment is locked.

## WaveDetect-adapted

Original WaveDetect processes token-probability signals from a surrogate language model using a continuous wavelet transform and spectral classifier. References are the [WaveDetect repository](https://github.com/KaitongQin/WaveDetect) and [authors' Hugging Face implementation](https://huggingface.co/KaitongQin/WaveDetect/blob/main/wavedetect_hf.py).

The package adaptation uses shared DeBERTa, up to eight chunks and a token-evidence head learned from the document objective. The centered token-probability signal is processed by a Morlet CWT with scales {1,2,4,8,16}, kernel 31, angular frequency 1.5, zero mean and L2 normalization. Log(1+CWT²) energy is summarized into 32 time bins and passed to a 1→8→16-channel CNN. The spectral projection is concatenated with mean CLS before LayerNorm and classification.

Differences in surrogate, signal, scales and classifier must be disclosed as adaptations. DeBERTa token-evidence probability is not causal next-token likelihood. The name WaveDetect-adapted is retained so readers can assess these differences.

## Binoculars

The ratio follows the orientation in the [official Binoculars code](https://github.com/ahans30/Binoculars/blob/main/binoculars/metrics.py): the numerator is mean next-token NLL from the performer; the denominator is observer-to-performer cross entropy at unshifted positions. It is not a ratio of two exponentiated perplexities.

Falcon-7B/Falcon-7B-Instruct use tokenizers checked for compatibility and at most 512 tokens. A smaller score indicates machine text. The threshold maximizes Macro-F1 on valid_select; one positive scale is fitted using NLL on valid_calib, allowing probability comparisons without changing the score threshold. This mapping is the package's evaluation choice, rather than direct application of a published BF16 threshold to quantized models.

The LMs are not fine-tuned. A single fixed evaluation is copied across the seed schedule for A5–Binoculars pairing. Binoculars summaries use n=1 with unavailable SD; pair counts in T09 represent A5 pipeline variation. The baseline scale fit is recorded in `Baseline_Fits`, separately from the 35 main A4 fits.

## Tables, statistics and external results

When all configurations and the external dataset are available, T09 follows the supplied generator: 207 internal contrasts and 42 external contrasts across 33 Family × Condition × Metric groups. These counts arise from available results; the script does not fill in unexecuted comparators.

External comparisons use A0, Longformer, Binoculars, DP-Net-adapted, WaveDetect-adapted, A4, Distance-Only and A5. Additional ablation comparators are outside this external protocol. Holm correction is applied within each family/condition/metric group; confidence intervals remain nominal.

The external CSV must contain text and source identities. Dataset version, sampling, domains/generators and filters must come from actual data and selection records. The five MAGE CSVs do not provide RAID. Without external input, T11 has no result rows and Fig11 is not generated.

Every table and figure is sourced from the training workbook. Statistics, timings, counts and checkpoints must be complete before new results are incorporated into the manuscript. Small-model software validation does not establish RIGAT, baseline or RAID benchmark scores.
