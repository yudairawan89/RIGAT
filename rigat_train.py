#!/usr/bin/env python3
"""Train RIGAT on released MAGE CSVs and export measured results to Excel.

Two entry points: this trainer and rigat_generate_actual.py (Excel -> figures).
Use --preflight before a long run. Checkpoints resume at accumulation boundaries.
"""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import train_test_split
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup
from safetensors.torch import save_file, load_file

import rigat_generate_actual as analysis

VERSION = "2.0.1"
METHODS = ["A0", "A1", "A2", "A2_DRO", "A3", "A4"]
AVAILABLE_METHODS = METHODS + ["Longformer", "DP-Net-adapted", "WaveDetect-adapted"]
PREFIX_METHODS = {"A0", "Longformer", "DP-Net-adapted"}
FILES = {"train": "train.csv", "valid": "valid.csv", "Standard": "test.csv",
         "OOD GPT": "test_ood_set_gpt.csv", "Paraphrase": "test_ood_set_gpt_para.csv"}
DOMAIN_NAMES = {"cmv": "CMV", "eli5": "ELI5", "hswag": "HellaSwag", "roct": "ROCStories",
    "squad": "SQuAD", "sci": "Scientific", "tldr": "TLDR", "wp": "WritingPrompts",
    "xsum": "XSum", "yelp": "Yelp", "cnn": "CNN", "imdb": "IMDb",
    "pubmed": "PubMed", "dialogsum": "DialogSum"}
GENERATOR_FAMILIES = ["OpenAI", "LLaMA", "OPT", "FLAN_T5", "GPT_Open", "OtherMachine"]


def dump_json(path, value):
    Path(path).write_text(json.dumps(analysis.json_safe(value), indent=2, allow_nan=False), encoding="utf-8")


def local_model_manifest(name):
    directory = Path(name)
    if not directory.is_dir():
        return None
    return {str(p.relative_to(directory)): analysis.sha256(p) for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix in {".json", ".txt", ".model", ".bin", ".safetensors"}}


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def family(src, label):
    if label == 1:
        return -1
    s = str(src).lower()
    if any(x in s for x in ["gpt-3", "gpt-4", "davinci"]):
        return 0
    if re.search(r"_(?:continuation|specified|topical)_(?:7|13|30|65)b$", s):
        return 1
    if "opt_" in s:
        return 2
    if "flan_t5" in s:
        return 3
    if any(x in s for x in ["gpt_j", "gpt_neox"]):
        return 4
    return 5


def subset(frame, maximum, seed):
    if not maximum or maximum >= len(frame):
        return frame.reset_index(drop=True)
    strata = frame.Domain + "|" + frame.label.astype(str)
    indices, _ = train_test_split(np.arange(len(frame)), train_size=maximum,
                                  random_state=seed, stratify=strata)
    return frame.iloc[np.sort(indices)].reset_index(drop=True)


def load_partitions(args, out, write=True):
    frames, manifests = {}, []
    for name, filename in FILES.items():
        path = args.data_dir / filename
        if not {"text", "label", "src"} <= set(pd.read_csv(path, nrows=0).columns):
            raise ValueError(f"{path} must contain text, label, src")
        frame = pd.read_csv(path, usecols=["src", "label"])
        if frame.isna().any().any() or not frame.label.isin([0, 1]).all():
            raise ValueError(f"Invalid labels or sources in {path}")
        frame["Row_Index"] = np.arange(len(frame))
        frame["Instance_ID"] = filename + ":" + frame.Row_Index.astype(str)
        prefix = frame.src.astype(str).str.split("_").str[0].str.lower()
        frame["Domain"] = prefix.map(DOMAIN_NAMES).fillna(prefix)
        frame["Generator_ID"] = [family(s, y) for s, y in zip(frame.src, frame.label)]
        manifests.append({"Split": name, "File": filename, "SHA256": analysis.sha256(path),
                          "Instances": len(frame), "Human": int(frame.label.sum()),
                          "Machine": int((frame.label == 0).sum()), "Domains": frame.Domain.nunique()})
        limit = args.max_train_docs if name == "train" else args.max_valid_docs if name == "valid" else args.max_test_docs
        frames[name] = subset(frame, limit, args.split_seed)
    domains = sorted(frames["train"].Domain.unique())
    domain_ids = {x: i for i, x in enumerate(domains)}
    for frame in frames.values():
        frame["Domain_ID"] = frame.Domain.map(domain_ids).fillna(-1).astype(int)
        frame["Group_ID"] = 2 * frame.Domain_ID + frame.label
    valid = frames.pop("valid")
    strata = valid.Domain + "|" + valid.label.astype(str)
    a, b = train_test_split(np.arange(len(valid)), test_size=round(.2 * len(valid)),
                           random_state=args.split_seed, stratify=strata)
    frames["valid_select"] = valid.iloc[np.sort(a)].reset_index(drop=True)
    frames["valid_calib"] = valid.iloc[np.sort(b)].reset_index(drop=True)
    if write:
        for name in ["train", "valid_select", "valid_calib"]:
            frames[name].to_csv(out / f"{name}_ids.csv", index=False)
        pd.DataFrame(manifests).to_csv(out / "input_manifest.csv", index=False)
    mapping = pd.DataFrame({"src": pd.concat(list(frames.values())).src.unique()})
    mapping["Domain"] = mapping.src.str.split("_").str[0].map(DOMAIN_NAMES)
    mapping["Machine_Family"] = [GENERATOR_FAMILIES[family(s, 0)] if "_human" not in s else None for s in mapping.src]
    if write:
        mapping.to_csv(out / "source_mapping.csv", index=False)
    if args.external_csv:
        if not args.external_protocol:
            raise ValueError("An external CSV requires --external-protocol with source/subset metadata")
        protocol = json.loads(args.external_protocol.read_text(encoding="utf-8"))
        for key in ["dataset_name", "dataset_version", "source_url", "selection_rule", "selection_seed"]:
            if key not in protocol or protocol[key] is None or str(protocol[key]) == "":
                raise ValueError(f"External protocol missing {key}")
        external = pd.read_csv(args.external_csv, usecols=["Instance_ID", "text", "label", "Domain", "Generator"])
        if external.isna().any().any() or external.Instance_ID.duplicated().any() or not external.label.isin([0, 1]).all():
            raise ValueError("External records need unique source IDs, text, 0/1 labels, domain and generator")
        external["Row_Index"] = np.arange(len(external))
        frames["External"] = external.drop(columns="text")
        protocol["csv_sha256"] = analysis.sha256(args.external_csv)
        if write:
            dump_json(out / "external_protocol.json", protocol)
    return frames, manifests, domain_ids


def save_partition_metadata(frames, manifests, args, out):
    for name in ["train", "valid_select", "valid_calib"]:
        frames[name].to_csv(out / f"{name}_ids.csv", index=False)
    pd.DataFrame(manifests).to_csv(out / "input_manifest.csv", index=False)
    sources = pd.concat([f[["src", "label"]] for f in frames.values() if "src" in f]).drop_duplicates()
    sources["Domain"] = sources.src.str.split("_").str[0].map(DOMAIN_NAMES)
    sources["Machine_Family"] = [GENERATOR_FAMILIES[family(s, y)] if y == 0 else None for s, y in zip(sources.src, sources.label)]
    sources.to_csv(out / "source_mapping.csv", index=False)
    if args.external_protocol:
        protocol = json.loads(args.external_protocol.read_text(encoding="utf-8"))
        protocol["csv_sha256"] = analysis.sha256(args.external_csv)
        dump_json(out / "external_protocol.json", protocol)


class TokenStore:
    """One tokenizer pass per source file; memory-mapped full token IDs across runs."""
    def __init__(self, path, tokenizer, cache_dir):
        self.tokenizer = tokenizer
        identity = hashlib.sha256((analysis.sha256(path) + tokenizer.backend_tokenizer.to_str()
                    if hasattr(tokenizer, "backend_tokenizer") else analysis.sha256(path) + str(tokenizer.get_vocab())).encode()).hexdigest()
        folder = cache_dir / identity[:24]
        folder.mkdir(parents=True, exist_ok=True)
        complete = folder / "complete.json"
        if not complete.exists():
            offsets = [0]
            with (folder / "tokens.tmp").open("wb") as stream:
                for frame in pd.read_csv(path, usecols=["text"], chunksize=64):
                    if frame.text.isna().any():
                        raise ValueError(f"Missing text in {path}")
                    batches = tokenizer(frame.text.astype(str).tolist(), add_special_tokens=False,
                                        truncation=False, return_attention_mask=False)["input_ids"]
                    for ids in batches:
                        values = np.asarray(ids, dtype=np.int32)
                        stream.write(values.tobytes())
                        offsets.append(offsets[-1] + len(values))
            (folder / "tokens.tmp").replace(folder / "tokens.dat")
            np.save(folder / "offsets.npy", np.asarray(offsets, dtype=np.int64))
            dump_json(complete, {"identity": identity, "rows": len(offsets)-1, "source": str(path.resolve())})
        info = json.loads(complete.read_text())
        if info["identity"] != identity:
            raise ValueError("Token-cache identity mismatch")
        self.offsets = np.load(folder / "offsets.npy", mmap_mode="r")
        self.ids = np.memmap(folder / "tokens.dat", mode="r", dtype=np.int32)

    def get(self, row):
        return self.ids[self.offsets[row]:self.offsets[row+1]].tolist()


def chunk_document(ids, tokenizer, length=512, overlap=64, maximum=8, prefix=False):
    special = tokenizer.num_special_tokens_to_add(pair=False)
    if special != 2 or tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("This implementation requires a CLS/SEP encoder tokenizer (DeBERTa or BERT)")
    payload, stride = length - special, length - overlap
    if payload < 1 or stride < 1 or stride > payload:
        raise ValueError("Invalid chunk length/overlap after reserving special tokens")
    n = max(1, math.ceil(max(0, len(ids)-payload)/stride) + 1)
    chosen = np.arange(min(1, n)) if prefix else np.arange(n) if n <= maximum else np.rint(np.linspace(0, n-1, maximum)).astype(int)
    starts = (chosen * stride).tolist()
    prepared = [[tokenizer.cls_token_id] + ids[s:s+payload] + [tokenizer.sep_token_id] for s in starts]
    intervals = sorted((s, min(s+payload, len(ids))) for s in starts)
    covered, end = 0, 0
    for a, b in intervals:
        covered += max(0, b-max(a, end))
        end = max(end, b)
    stats = {"Total_Content_Tokens": len(ids), "Available_Token_Chunks": n,
             "Selected_Token_Chunks": len(prepared), "Covered_Content_Tokens": covered,
             "Token_Coverage": covered/len(ids) if len(ids) else 1., "Complete_Document": int(covered == len(ids))}
    return prepared, stats


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -grad


class RIGAT(nn.Module):
    def __init__(self, backbone, method, n_domains, max_chunks=8, dropout=.1, checkpointing=True,
                 dp_scales=(0., .01, .03, .06, .1), wave_scales=(1., 2., 4., 8., 16.), wave_kernel=31, wave_bins=32):
        super().__init__()
        self.encoder = backbone
        self.method = method
        self.reliability = method in {"A2", "A2_DRO", "A3", "A4"}
        self.invariant = method in {"A3", "A4"}
        self.robust = method in {"A2_DRO", "A4"}
        h = backbone.config.hidden_size
        self.positions = nn.Embedding(max_chunks, h) if method not in PREFIX_METHODS else None
        self.normalization = nn.LayerNorm(h) if method not in PREFIX_METHODS else nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, 1))
        if self.reliability:
            self.local_head = nn.Sequential(nn.Linear(h, h//2), nn.Tanh(), nn.Dropout(dropout), nn.Linear(h//2, 1))
            self.score_head = nn.Sequential(nn.Linear(h, h//2), nn.Tanh(), nn.Linear(h//2, 1))
            self.gamma_raw = nn.Parameter(torch.tensor(0.0))
        if self.invariant:
            self.domain_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, n_domains))
            self.generator_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h, len(GENERATOR_FAMILIES)))
        self.dp_aux = None
        if method == "DP-Net-adapted":
            self.dp_extractor = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Dropout(dropout))
            self.dp_policy = nn.Sequential(nn.Linear(h, 32), nn.Tanh(), nn.Linear(32, len(dp_scales)))
            self.register_buffer("dp_scales", torch.tensor(dp_scales))
            self.register_buffer("dp_reward_baseline", torch.tensor(0.))
        if method == "WaveDetect-adapted":
            self.wave_token_head = nn.Linear(h, 1)
            time_axis = torch.arange(wave_kernel, dtype=torch.float32)-(wave_kernel-1)/2
            kernels = []
            for scale in wave_scales:
                kernel = torch.exp(-.5*(time_axis/scale)**2)*torch.cos(1.5*time_axis/scale)
                kernel = kernel-kernel.mean()
                kernels.append(kernel/kernel.norm().clamp_min(1e-8))
            self.register_buffer("wave_kernels", torch.stack(kernels)[:, None, :])
            self.wave_bins = wave_bins
            self.wave_cnn = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
            self.wave_projection = nn.Linear(16, h)
        if checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def forward(self, chunks, chunk_batch=1):
        vectors, wave_vectors = [], []
        for start in range(0, len(chunks["input_ids"]), chunk_batch):
            inputs = {k: v[start:start+chunk_batch] for k, v in chunks.items()}
            if self.encoder.config.model_type == "longformer":
                inputs["global_attention_mask"] = torch.zeros_like(inputs["attention_mask"])
                inputs["global_attention_mask"][:, 0] = 1
            hidden = self.encoder(**inputs).last_hidden_state
            vectors.append(hidden[:, 0])
            if self.method == "WaveDetect-adapted":
                for i in range(len(hidden)):
                    count = int(inputs["attention_mask"][i].sum())
                    # Keep small spectral energies in FP32 under encoder AMP.
                    with torch.autocast(device_type=hidden.device.type, enabled=False):
                        signal = torch.sigmoid(self.wave_token_head(hidden[i, 1:max(2, count-1)].float()).squeeze(-1))
                        signal = signal-signal.mean()
                        spectrum = F.conv1d(signal[None, None, :], self.wave_kernels.float(), padding=self.wave_kernels.shape[-1]//2)
                        energy = F.adaptive_avg_pool1d(torch.log1p(spectrum.square()), self.wave_bins)
                        feature = self.wave_cnn(energy[:, None, :, :]).flatten(1)
                        wave_vectors.append(self.wave_projection(feature).squeeze(0))
        h = torch.cat(vectors)
        if self.positions is not None:
            h = h + self.positions(torch.arange(len(h), device=h.device))
        local_logits = None
        if self.reliability:
            local_logits = self.local_head(h).squeeze(-1)
            p = torch.sigmoid(local_logits.float()).clamp(1e-7, 1-1e-7)
            entropy = -(p*p.log() + (1-p)*torch.log1p(-p))/math.log(2)
            weights = torch.softmax(self.score_head(h).squeeze(-1).float()
                        + F.softplus(self.gamma_raw.float()) * torch.log(1-entropy+1e-6), dim=0)
            z = self.normalization((weights[:, None]*h.float()).sum(0))
        else:
            z = h.float().mean(0)
            if self.method != "WaveDetect-adapted":
                z = self.normalization(z)
        if self.method == "WaveDetect-adapted":
            z = self.normalization(z+torch.stack(wave_vectors).float().mean(0))
        if self.method == "DP-Net-adapted":
            original = z
            z = self.dp_extractor(original)
            logit = self.classifier(z).squeeze()
            self.dp_aux = None
            if self.training:
                policy = torch.distributions.Categorical(logits=self.dp_policy(original.detach()).float())
                action = policy.sample()
                noise = torch.randn_like(original)*self.dp_scales[action]*original.detach().square().mean().sqrt().clamp_min(1e-6)
                noisy = self.dp_extractor(original+noise)
                self.dp_aux = {"logit": self.classifier(noisy).squeeze(), "alignment": (z.float()-noisy.float()).square().mean(),
                               "log_probability": policy.log_prob(action), "entropy": policy.entropy()}
            return logit, z, local_logits
        return self.classifier(z).squeeze(), z, local_logits


def device_chunks(prepared, tokenizer, device):
    maximum = max(map(len, prepared))
    ids = [p+[tokenizer.pad_token_id]*(maximum-len(p)) for p in prepared]
    masks = [[1]*len(p)+[0]*(maximum-len(p)) for p in prepared]
    return {"input_ids": torch.tensor(ids, device=device), "attention_mask": torch.tensor(masks, device=device)}


def autocast(args, device):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" and args.precision == "fp16" else nullcontext()


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak(device):
    if device.type != "cuda":
        return {"Peak_Allocated_VRAM_GB": None, "Peak_Reserved_VRAM_GB": None}
    return {"Peak_Allocated_VRAM_GB": torch.cuda.max_memory_allocated(device)/1024**3,
            "Peak_Reserved_VRAM_GB": torch.cuda.max_memory_reserved(device)/1024**3}


def inference(model, frame, store, tokenizer, args, device, k=None, embeddings=False):
    model.eval()
    rows, representations = [], []
    with torch.inference_mode():
        for row in frame.itertuples():
            prepared, coverage = chunk_document(store.get(row.Row_Index), tokenizer,
                args.chunk_length, args.overlap, k or args.max_chunks, model.method in PREFIX_METHODS)
            inputs = device_chunks(prepared, tokenizer, device)
            with autocast(args, device):
                logit, z, _ = model(inputs, args.chunk_batch)
            if not torch.isfinite(logit) or not torch.isfinite(z).all():
                raise FloatingPointError(f"Nonfinite inference output for {row.Instance_ID}")
            rows.append({"Instance_ID": row.Instance_ID, "label": row.label, "Domain": row.Domain,
                         "Raw_Logit": float(logit.float().cpu()), **coverage})
            if embeddings:
                representations.append(z.float().cpu().numpy())
    return pd.DataFrame(rows), np.asarray(representations) if embeddings else None


def distance_values(z, reference):
    values = []
    for mean in reference["means"]:
        delta = np.asarray(z, dtype=np.float64) - mean
        values.append(np.maximum(0, np.einsum("ij,jk,ik->i", delta, reference["precision"], delta)))
    return np.min(values, axis=0)/float(reference["scale"])


def fit_reference(z, y, normalization="none"):
    means = np.stack([z[y == c].mean(0) for c in [0, 1]])
    centered = z-means[y]
    precision = LedoitWolf().fit(centered).precision_
    reference = {"means": means, "precision": precision, "scale": 1.0}
    if normalization == "train-median":
        reference["scale"] = max(float(np.median(distance_values(z, reference))), 1e-12)
    return reference


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train_stage_one(model, frame, select, stores, tokenizer, args, device, run_dir, seed, identity):
    model.to(device)
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    heads = [p for n, p in model.named_parameters() if not n.startswith("encoder.") and p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": encoder_params, "lr": args.encoder_lr},
                                   {"params": heads, "lr": args.head_lr}], weight_decay=args.weight_decay)
    n_steps = math.ceil(len(frame)/args.accumulation)*args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, round(n_steps*args.warmup_ratio), n_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    y = frame.label.to_numpy()
    weights = torch.tensor([len(y)/(2*(y == c).sum()) for c in [0, 1]], device=device, dtype=torch.float32)
    groups = frame.Group_ID.to_numpy(dtype=int)
    active = np.unique(groups)
    log_q = torch.full((int(groups.max())+1,), -float("inf"), device=device)
    log_q[torch.tensor(active, device=device)] = -math.log(len(active))
    epoch, cursor, step, elapsed, history, last_norm = 0, 0, 0, 0.0, [], float("nan")
    successful_updates = 0
    checkpoint = run_dir / "resume.pt"
    train_peak = {"Peak_Allocated_VRAM_GB": None, "Peak_Reserved_VRAM_GB": None}
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("Resume configuration/dataset differs from saved checkpoint")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        log_q = state["log_q"].to(device)
        epoch, cursor, step, elapsed, history = [state[k] for k in ["epoch", "cursor", "step", "elapsed", "history"]]
        train_peak = state["peak"]
        last_norm = state["last_norm"]
        successful_updates = state.get("successful_updates", step)
        restore_rng(state["rng"])
        del state
    def save(epoch_value, cursor_value):
        temp = checkpoint.with_suffix(".tmp")
        torch.save({"identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "log_q": log_q,
            "epoch": epoch_value, "cursor": cursor_value, "step": step, "elapsed": elapsed,
            "history": history, "peak": train_peak, "last_norm": last_norm,
            "successful_updates": successful_updates, "rng": rng_state()}, temp)
        temp.replace(checkpoint)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for e in range(epoch, args.epochs):
        order = np.random.default_rng(seed + 100003*e).permutation(len(frame))
        start_cursor = cursor if e == epoch else 0
        model.train()
        for offset in range(start_cursor, len(order), args.accumulation):
            indices = order[offset:offset+args.accumulation]
            counts = np.bincount(groups[indices], minlength=len(log_q))
            machine_count = max(1, int((y[indices] == 0).sum()))
            risks = np.zeros(len(log_q))
            q = log_q.exp().detach()
            sync(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            for index in indices:
                row = frame.iloc[int(index)]
                prepared, _ = chunk_document(stores["train"].get(int(row.Row_Index)), tokenizer,
                    args.chunk_length, args.overlap, args.max_chunks, model.method in PREFIX_METHODS)
                inputs = device_chunks(prepared, tokenizer, device)
                target = torch.tensor(float(row.label), device=device)
                with autocast(args, device):
                    logit, z, local = model(inputs, args.chunk_batch)
                    cb = F.binary_cross_entropy_with_logits(logit.float(), target)*weights[int(row.label)]
                    loss = cb
                    if model.method == "DP-Net-adapted":
                        aux = model.dp_aux
                        noisy_cb = F.binary_cross_entropy_with_logits(aux["logit"].float(), target)*weights[int(row.label)]
                        loss = .5*cb+.5*noisy_cb+args.dp_alignment*aux["alignment"]
                        reward = -loss.detach()
                        advantage = reward-model.dp_reward_baseline.detach().clone()
                        loss = loss+args.dp_policy_weight*(-advantage*aux["log_probability"]-args.dp_entropy*aux["entropy"])
                        with torch.no_grad():
                            model.dp_reward_baseline.mul_(.95).add_(.05*reward)
                    if local is not None:
                        loss = loss + args.lambda_local*F.binary_cross_entropy_with_logits(
                            local.float(), target.expand_as(local))
                    if model.invariant:
                        reverse = ReverseGradient.apply(z)
                        inv = F.cross_entropy(model.domain_head(reverse).unsqueeze(0),
                            torch.tensor([int(row.Domain_ID)], device=device))
                        if row.label == 0:
                            inv = inv + len(indices)/machine_count*F.cross_entropy(
                                model.generator_head(reverse).unsqueeze(0),
                                torch.tensor([int(row.Generator_ID)], device=device))
                        loss = loss + args.lambda_invariant*inv
                    if model.robust:
                        g = int(row.Group_ID)
                        loss = loss + args.lambda_robust*len(indices)*q[g]/int(counts[g])*cb
                    loss = loss/len(indices)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite Stage I loss: seed={seed}, method={model.method}, step={step}")
                scaler.scale(loss).backward()
                risks[int(row.Group_ID)] += float(cb.detach())
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            last_norm = float(norm)
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            successful_update = scaler.get_scale() >= previous_scale
            if successful_update:
                scheduler.step()
                successful_updates += 1
            if model.robust:
                observed = np.flatnonzero(counts)
                with torch.no_grad():
                    ix = torch.tensor(observed, device=device)
                    log_q[ix] += args.group_eta*torch.tensor(risks[observed]/counts[observed], device=device)
                    log_q -= torch.logsumexp(log_q, dim=0)
            sync(device)
            elapsed += time.perf_counter()-start
            current_peak = peak(device)
            for key, value in current_peak.items():
                if value is not None:
                    train_peak[key] = max(train_peak[key] or 0, value)
            step += 1
            if step % args.log_every == 0:
                print(f"  {model.method} seed {seed} epoch {e+1}/{args.epochs}: {offset+len(indices)}/{len(order)} docs, step {step}", flush=True)
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                save(e, offset+len(indices))
        prediction, _ = inference(model, select, stores["valid"], tokenizer, args, device)
        met = analysis.metrics(prediction.label.to_numpy(), prediction.Raw_Logit.to_numpy(), curves=False)[0]
        history.append({"Seed": seed, "Method": model.method, "Epoch": e+1, "Accumulation_Windows": step,
                        "Optimizer_Updates": successful_updates, "AMP_Skipped_Updates": step-successful_updates,
                        "Training_Elapsed_Sec": elapsed, "Selection_Macro_F1": met["Macro_F1"],
                        "Selection_NLL": met["NLL"], "Gradient_Norm_Last": last_norm})
        save(e+1, 0)
        cursor = 0
        print(f"  {model.method} seed {seed}: epoch {e+1} complete; selection F1={met['Macro_F1']:.5f}", flush=True)
    model.eval()
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(run_dir / "model.safetensors"))
    del optimizer, scheduler, scaler
    model.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return elapsed, train_peak, pd.DataFrame(history)


def benchmark(model, text_rows, tokenizer, args, device, seed, method, reference=None, calibrator=None, k=None, series="main"):
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    records = []
    begin, end = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) if device.type == "cuda" else (None, None)
    with torch.inference_mode():
        for i in range(args.timing_warmup + args.timing_runs):
            instance_id, text = text_rows[i % len(text_rows)]
            sync(device)
            start = time.perf_counter()
            ids = tokenizer(str(text), add_special_tokens=False, truncation=False)["input_ids"]
            prepared, coverage = chunk_document(ids, tokenizer, args.chunk_length, args.overlap,
                k or args.max_chunks, model.method in PREFIX_METHODS)
            inputs = device_chunks(prepared, tokenizer, device)
            if begin is not None:
                begin.record()
            with autocast(args, device):
                logit, z, _ = model(inputs, args.chunk_batch)
            if end is not None:
                end.record()
            value = float(logit.float().cpu())
            if calibrator is not None:
                distance = distance_values(z.float().cpu().numpy()[None, :], reference)
                value = float(calibrator.transform(np.array([value]), analysis.raw_entropy(np.array([value])), distance)[0])
            human_probability = float(analysis.probability(np.array([value]))[0])
            sync(device)
            wall_ms = 1000*(time.perf_counter()-start)
            if i >= args.timing_warmup:
                records.append({"Seed": seed, "Method": method, "Kmax": k or args.max_chunks,
                    "Timing_ID": f"{seed}:{method}:K{k or args.max_chunks}:{series}:{i-args.timing_warmup}",
                    "Timing_Run": i-args.timing_warmup, "Instance_ID": instance_id,
                    "End_to_End_ms": wall_ms,
                    "CUDA_Model_ms": begin.elapsed_time(end) if begin is not None else np.nan,
                    "Selected_Token_Chunks": coverage["Selected_Token_Chunks"],
                    "Human_Probability": human_probability})
    return pd.DataFrame(records), peak(device)


def timing_texts(args, standard, seed):
    rng = np.random.default_rng(seed + 523)
    chosen = standard.iloc[rng.permutation(len(standard))[:min(len(standard), args.timing_runs+args.timing_warmup)]]
    needed = dict(zip(chosen.Row_Index.astype(int), chosen.Instance_ID))
    texts, row_offset = {}, 0
    for block in pd.read_csv(args.data_dir / "test.csv", usecols=["text"], chunksize=2000):
        for index, text in enumerate(block.text, row_offset):
            if index in needed:
                texts[index] = text
        row_offset += len(block)
    return [(row.Instance_ID, texts[int(row.Row_Index)]) for row in chosen.itertuples()]


def performance_records(frame, seed, method, condition, calibrator=None):
    y, logits = frame.label.to_numpy(), frame.Logit.to_numpy()
    met, bins, grid, risk = analysis.metrics(y, logits)
    identifier = {"Seed": seed, "Condition": condition, "Method": method}
    if calibrator is not None:
        identifier["Calibrator"] = calibrator
    domains = []
    for domain, group in frame.groupby("Domain", sort=True):
        local = analysis.metrics(group.label.to_numpy(), group.Logit.to_numpy(), curves=False)[0]
        domains.append({**identifier, "Domain": domain, "N": len(group), **local})
    return ({**identifier, "N": len(frame), **met},
            [{**identifier, **b} for b in bins],
            [{**identifier, "Coverage": float(c), "Selective_Risk": float(r)} for c, r in zip(grid, risk)], domains)


def run_evaluation(model, frames, stores, tokenizer, args, device, run_dir, seed, standard_words):
    records = {name: [] for name in ["Classification", "Calibration", "Bins", "Risk", "Domain", "Coverage",
        "Calibration_Diagnosis", "Coverage_Fits", "Validation_Logits", "Timing", "Coverage_Timing", "ROC_PR", "Cost"]}
    predictions = []
    def save_prediction(frame, name):
        path = run_dir / (name + ".csv.gz")
        frame.to_csv(path, index=False, compression="gzip")
        predictions.append({"Seed": seed, "Method": model.method, "File": str(path),
                            "SHA256": analysis.sha256(path), "Instances": len(frame)})
    def add_evaluation(frame, method, condition, name=None):
        result = performance_records(frame, seed, method, condition, name)
        records["Calibration" if name else "Classification"].append(result[0])
        if name:
            records["Bins"].extend(result[1])
            records["Risk"].extend(result[2])
        else:
            records["Domain"].extend(result[3])
            fpr, tpr, _ = analysis.roc_curve(1-frame.label.to_numpy(), -frame.Logit.to_numpy())
            precision, recall, _ = analysis.precision_recall_curve(1-frame.label.to_numpy(), -frame.Logit.to_numpy())
            grid = np.linspace(0, 1, 201)
            records["ROC_PR"].extend({"Seed": seed, "Method": method, "Condition": condition,
                "Grid": float(x), "ROC_TPR": float(t), "PR_Precision": float(p)}
                for x, t, p in zip(grid, np.interp(grid, fpr, tpr), np.interp(grid, recall[::-1], precision[::-1])))
    reference, calibrators, val, coverage_fits = None, {}, None, {}
    stage_times = {}
    if model.method == "A4":
        start = time.perf_counter()
        selected = frames["train"].iloc[np.random.default_rng(seed+919).permutation(len(frames["train"]))[:args.reference_docs]]
        ref_frame, ref_z = inference(model, selected, stores["train"], tokenizer, args, device, embeddings=True)
        stage_times["Reference_Encoding_Wall_Sec"] = time.perf_counter()-start
        start = time.perf_counter()
        reference = fit_reference(ref_z, ref_frame.label.to_numpy(dtype=int), args.distance_normalization)
        stage_times["Reference_Statistics_Wall_Sec"] = time.perf_counter()-start
        np.savez_compressed(run_dir / "reference.npz", **reference)
        selected.to_csv(run_dir / "reference_ids.csv", index=False)
        start = time.perf_counter()
        val, z = inference(model, frames["valid_calib"], stores["valid"], tokenizer, args, device, embeddings=True)
        stage_times["Calibration_Input_Encoding_Wall_Sec"] = time.perf_counter()-start
        start = time.perf_counter()
        val["Distance"] = distance_values(z, reference)
        val["Entropy"] = analysis.raw_entropy(val.Raw_Logit.to_numpy())
        instance = (val.label.to_numpy(), val.Raw_Logit.to_numpy(), val.Entropy.to_numpy(), val.Distance.to_numpy())
        stage_times["Calibration_Features_Wall_Sec"] = time.perf_counter()-start
        calibrators = {name: analysis.fit_calibrator(name, *instance, maxiter=args.calibration_maxiter,
                        trust_l2=args.trust_l2) for name in analysis.CALIBRATORS}
        if any(not c.fit_info.get("Success", True) for c in calibrators.values()):
            print("Calibration convergence is incomplete; inspect Calibration_Diagnosis optimizer status.", flush=True)
        records["Calibration_Diagnosis"] = analysis.calibration_diagnosis(seed, calibrators, instance)
        validation_export = val.copy()
        validation_export["Seed"] = seed
        for name, cal in calibrators.items():
            validation_export[name] = cal.transform(*instance[1:])
        records["Validation_Logits"] = validation_export.to_dict("records")
        dump_json(run_dir / "calibrators.json", {name: c.serializable() for name, c in calibrators.items()})
        if not args.skip_coverage:
            for k in args.coverage_k:
                if k == args.max_chunks:
                    coverage_fits[k] = calibrators["Adaptive Trust (A5)"]
                    continue
                restricted, z = inference(model, frames["valid_calib"], stores["valid"], tokenizer, args, device, k, True)
                restricted["Distance"] = distance_values(z, reference)
                restricted["Entropy"] = analysis.raw_entropy(restricted.Raw_Logit.to_numpy())
                data = (restricted.label.to_numpy(), restricted.Raw_Logit.to_numpy(),
                        restricted.Entropy.to_numpy(), restricted.Distance.to_numpy())
                coverage_fits[k] = analysis.fit_calibrator("Adaptive Trust (A5)", *data,
                    maxiter=args.calibration_maxiter, trust_l2=args.trust_l2)
                diagnosis = analysis.calibration_diagnosis(seed, {"Adaptive Trust (A5)": coverage_fits[k]}, data)[0]
                records["Coverage_Fits"].append({"Kmax": k, **diagnosis})
                save_prediction(restricted, f"coverage_calibration_K{k}")
            dump_json(run_dir / "coverage_calibrators.json", {str(k): c.serializable() for k, c in coverage_fits.items()})
    external_methods = {"A0", "Longformer", "DP-Net-adapted", "WaveDetect-adapted", "A4"}
    for condition in ["Standard", "OOD GPT", "Paraphrase"] + (["External"] if "External" in frames and model.method in external_methods else []):
        frame, z = inference(model, frames[condition], stores[condition], tokenizer, args, device,
                              embeddings=model.method == "A4")
        frame["Seed"], frame["Method"] = seed, model.method
        frame["Logit"] = frame.Raw_Logit
        frame["Entropy"] = analysis.raw_entropy(frame.Raw_Logit.to_numpy())
        if reference is not None:
            frame["Distance"] = distance_values(z, reference)
        add_evaluation(frame, model.method, condition)
        save_prediction(frame, condition.replace(" ", "_"))
        if calibrators:
            for name, cal in calibrators.items():
                mapped = frame.copy()
                mapped["Logit"] = cal.transform(frame.Raw_Logit.to_numpy(), frame.Entropy.to_numpy(), frame.Distance.to_numpy())
                add_evaluation(mapped, "A4", condition, name)
                if name == "Adaptive Trust (A5)":
                    if not np.array_equal(mapped.Logit >= 0, frame.Raw_Logit >= 0):
                        raise AssertionError("Positive trust scaling changed a decision")
                    mapped["Method"] = "A5"
                    mapped["Trust"] = analysis.expit(cal.parameters["Bias"]-cal.parameters["Alpha"]*frame.Entropy.to_numpy()
                                                   -cal.parameters["Beta"]*frame.Distance.to_numpy())
                    add_evaluation(mapped, "A5", condition)
                    save_prediction(mapped, condition.replace(" ", "_")+"_A5")
                elif condition == "External" and name == "Distance-Only":
                    mapped["Method"] = "A4_Distance_Only"
                    add_evaluation(mapped, "A4_Distance_Only", condition)
                    save_prediction(mapped, "External_Distance_Only")
            if condition == "Standard" and not args.skip_coverage:
                full_z = z
                for k in args.coverage_k:
                    restricted, z = (frame.copy(), full_z) if k == args.max_chunks else inference(
                        model, frames[condition], stores[condition], tokenizer, args, device, k, True)
                    restricted["Distance"] = distance_values(z, reference)
                    restricted["Entropy"] = analysis.raw_entropy(restricted.Raw_Logit.to_numpy())
                    save_prediction(restricted, f"coverage_raw_K{k}")
                    for protocol, cal in [(f"Fixed K={args.max_chunks} calibration", calibrators["Adaptive Trust (A5)"]),
                                           ("K-specific calibration", coverage_fits[k])]:
                        scores = cal.transform(restricted.Raw_Logit.to_numpy(), restricted.Entropy.to_numpy(), restricted.Distance.to_numpy())
                        met = analysis.metrics(restricted.label.to_numpy(), scores)[0]
                        records["Coverage"].append({"Seed": seed, "Protocol": protocol, "Kmax": k,
                            "N": len(restricted), "Token_Coverage_Mean": restricted.Token_Coverage.mean(),
                            "Complete_Token_Document_Fraction": restricted.Complete_Document.mean(),
                            "Observed_Token_Chunks_Mean": restricted.Selected_Token_Chunks.mean(),
                            "Words_Per_Descriptive_Chunk": 512,
                            "Observed_512Word_Chunks_Mean": np.minimum(np.maximum(1, (standard_words+511)//512), k).mean(),
                            "Word_Coverage_Mean": np.where(standard_words > 0, np.minimum(standard_words, k*512)/np.maximum(1, standard_words), 1).mean(),
                            "Complete_Word_Document_Fraction": (standard_words <= k*512).mean(), **met})
    text_rows = timing_texts(args, frames["Standard"], seed)
    timing, infer_peak = benchmark(model, text_rows, tokenizer, args, device, seed, model.method)
    records["Timing"].extend(timing.to_dict("records"))
    resources = [{"Seed": seed, "Method": model.method, "Independent_Training_Run_Count": 1,
                  "Calibration_Fit_CPU_Sec": np.nan, **infer_peak}]
    if calibrators:
        timing_a5, infer_a5 = benchmark(model, text_rows, tokenizer, args, device, seed, "A5", reference,
                                         calibrators["Adaptive Trust (A5)"])
        records["Timing"].extend(timing_a5.to_dict("records"))
        a5_fit = next(r["Fit_CPU_Sec"] for r in records["Calibration_Diagnosis"] if r["Calibrator"] == "Adaptive Trust (A5)")
        resources.append({"Seed": seed, "Method": "A5", "Independent_Training_Run_Count": 0,
                          "Parent_Stage_I": "A4", "Calibration_Fit_CPU_Sec": a5_fit,
                          "Stage_II_A5_Compute_Wall_Sec": sum(stage_times.values())+a5_fit, **stage_times, **infer_a5})
        if not args.skip_coverage:
            for k in args.coverage_k:
                for protocol, cal in [(f"Fixed K={args.max_chunks} calibration", calibrators["Adaptive Trust (A5)"]),
                                       ("K-specific calibration", coverage_fits[k])]:
                    measured = timing_a5.copy() if k == args.max_chunks else benchmark(model, text_rows, tokenizer, args,
                        device, seed, "A5", reference, cal, k, series=protocol)[0]
                    measured["Kmax"], measured["Protocol"] = k, protocol
                    records["Coverage_Timing"].extend(measured.to_dict("records"))
    return records, resources, predictions


def hardware(device):
    result = {"Execution_Device": str(device), "CPU": platform.processor(), "System_RAM_GB": None,
              "GPU": None, "GPU_VRAM_GB": None}
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                result["CPU"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        import psutil
        result["System_RAM_GB"] = psutil.virtual_memory().total/1024**3
    except ImportError:
        pass
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["GPU"], result["GPU_VRAM_GB"] = properties.name, properties.total_memory/1024**3
    return result


def binoculars_ratio(observer_logits, performer_logits, ids, mask, pad_id):
    """Reference orientation: performer shifted NLL / unshifted O->P cross entropy.

    https://github.com/ahans30/Binoculars/blob/main/binoculars/metrics.py
    Vocab distributions are compared in blocks to bound host temporary memory.
    """
    observer = observer_logits.float().cpu()[0]
    performer = performer_logits.float().cpu()[0]
    tokens, attention = ids.cpu()[0], mask.cpu()[0]
    if len(tokens) < 2 or attention[1:].sum() == 0:
        raise ValueError("Binoculars requires at least two observed tokens")
    numerator = (F.cross_entropy(performer[:-1], tokens[1:], reduction="none")*attention[1:]).sum()/attention[1:].sum()
    total, count = 0., 0
    for start in range(0, len(tokens), 32):
        end = min(start+32, len(tokens))
        valid = tokens[start:end] != pad_id
        cross_entropy = -(torch.softmax(observer[start:end], dim=-1)*torch.log_softmax(performer[start:end], dim=-1)).sum(-1)
        total += float(cross_entropy[valid].sum())
        count += int(valid.sum())
    if count == 0 or not math.isfinite(total) or total <= 0:
        raise ValueError("Invalid Binoculars cross-entropy denominator")
    return float(numerator)/(total/count)


class BinocularsRunner:
    def __init__(self, args, config, device, folder):
        self.args, self.device = args, device
        references = config["binoculars_references"]
        names = [args.binoculars_observer, args.binoculars_performer]
        self.tokenizer = AutoTokenizer.from_pretrained(names[0], revision=references[0]["revision"], local_files_only=args.local_files_only)
        performer_tokenizer = AutoTokenizer.from_pretrained(names[1], revision=references[1]["revision"], local_files_only=args.local_files_only)
        if self.tokenizer.get_vocab() != performer_tokenizer.get_vocab():
            raise ValueError("Binoculars observer/performer token vocabularies differ")
        for probe in ["A short text.", "Unicode: café!", "Numbers 123 and punctuation."]:
            if self.tokenizer.encode(probe) != performer_tokenizer.encode(probe):
                raise ValueError("Binoculars tokenizer preprocessing differs")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.models = []
        for index, name in enumerate(names):
            kwargs = {"revision": references[index]["revision"], "local_files_only": args.local_files_only}
            if device.type == "cuda":
                try:
                    import bitsandbytes
                except ImportError as exc:
                    raise RuntimeError("Binoculars GPU scoring requires requirements_binoculars_gpu.txt") from exc
                kwargs.update({"device_map": "auto", "torch_dtype": torch.float16,
                    "quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16, llm_int8_enable_fp32_cpu_offload=True),
                    "max_memory": {0: int(args.binoculars_gpu_budget_gb*1024**3), "cpu": int(args.binoculars_cpu_budget_gb*1024**3)},
                    "offload_folder": str(folder / f"offload_{index}")})
            else:
                kwargs["torch_dtype"] = torch.float32
            model = AutoModelForCausalLM.from_pretrained(name, **kwargs).eval()
            if device.type == "cpu":
                model.to(device)
            dump_json(folder / f"model_reference_{index}.json", {**references[index],
                "config": model.config.to_dict(), "device_map": {str(k): str(v) for k, v in getattr(model, "hf_device_map", {}).items()}})
            self.models.append(model)
        if self.models[0].config.vocab_size != self.models[1].config.vocab_size:
            raise ValueError("Binoculars vocabulary sizes differ")
        self.events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(2)] if device.type == "cuda" else []

    @torch.inference_mode()
    def score(self, text, measure=False):
        encoded = self.tokenizer(str(text), return_tensors="pt", truncation=True,
            max_length=self.args.binoculars_max_tokens, return_token_type_ids=False)
        logits, elapsed = [], 0.
        for index, model in enumerate(self.models):
            target = model.device if model.device.type != "meta" else torch.device("cpu")
            inputs = {k: v.to(target) for k, v in encoded.items() if k in {"input_ids", "attention_mask"}}
            if measure and self.events:
                self.events[index][0].record()
            values = model(**inputs).logits
            if measure and self.events:
                self.events[index][1].record()
                sync(self.device)
                elapsed += self.events[index][0].elapsed_time(self.events[index][1])
            logits.append(values.detach().float().cpu())
            del values
        score = binoculars_ratio(logits[0], logits[1], encoded["input_ids"], encoded["attention_mask"], self.tokenizer.pad_token_id)
        return score, elapsed if self.events and measure else np.nan, int(encoded["attention_mask"].sum())


def load_text_lookup(args, name, frame):
    path = args.external_csv if name == "External" else args.data_dir / ("valid.csv" if name.startswith("valid_") else FILES[name])
    needed = set(frame.Row_Index.astype(int))
    lookup, offset = {}, 0
    for block in pd.read_csv(path, usecols=["text"], chunksize=2000):
        for i, value in enumerate(block.text, offset):
            if i in needed:
                lookup[i] = value
        offset += len(block)
    return lookup


def select_score_threshold(y, scores):
    order = np.argsort(scores, kind="stable")
    s, labels = np.asarray(scores)[order], np.asarray(y)[order]
    ends = np.r_[0, np.flatnonzero(np.diff(s) != 0)+1, len(s)]
    human = np.r_[0, np.cumsum(labels == 1)]
    fp = human[ends]
    tp = ends-fp
    fn, tn = (labels == 0).sum()-tp, (labels == 1).sum()-fp
    macro = .5*(2*tp/np.maximum(1, 2*tp+fn+fp)+2*tn/np.maximum(1, 2*tn+fn+fp))
    cut = int(ends[np.argmax(macro)])
    if cut == 0:
        return float(np.nextafter(s[0], -np.inf))
    if cut == len(s):
        return float(np.nextafter(s[-1], np.inf))
    return float(s[cut-1]+(s[cut]-s[cut-1])/2)


def run_binoculars(args, config, frames, device, out):
    folder = out / "runs/frozen_Binoculars"
    folder.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(json.dumps({"config": config, "method": "Binoculars"}, sort_keys=True).encode()).hexdigest()
    complete = folder / "run_complete.json"
    if complete.exists():
        if not args.resume or json.loads(complete.read_text())["identity"] != identity:
            raise ValueError("Completed Binoculars run exists; use matching --resume")
        return
    progress = folder / "progress_identity.json"
    if progress.exists():
        if not args.resume or json.loads(progress.read_text())["identity"] != identity:
            raise ValueError("Binoculars scoring progress exists; use matching --resume")
    else:
        dump_json(progress, {"identity": identity})
    seed_all(2026)
    runner = BinocularsRunner(args, config, device, folder)
    def predict(name):
        frame = frames[name]
        path = folder / (name.replace(" ", "_")+"_scores.csv.gz")
        partial = folder / (name.replace(" ", "_")+"_partial.csv")
        if args.resume and path.exists():
            cached = pd.read_csv(path)
            if not cached.Instance_ID.astype(str).equals(frame.Instance_ID.astype(str)) or not np.array_equal(cached.label, frame.label) or not np.isfinite(cached.Binoculars_Score).all():
                raise ValueError("Cached Binoculars population or scores are invalid")
            return cached
        cached = pd.read_csv(partial) if args.resume and partial.exists() else pd.DataFrame()
        if not cached.empty:
            if not cached.Instance_ID.astype(str).equals(frame.Instance_ID.iloc[:len(cached)].astype(str).reset_index(drop=True)) or not np.array_equal(cached.label, frame.label.iloc[:len(cached)]) or not np.isfinite(cached.Binoculars_Score).all():
                raise ValueError("Partial Binoculars score cache is invalid")
        lookup = load_text_lookup(args, name, frame)
        rows = []
        for number, row in enumerate(frame.iloc[len(cached):].itertuples(), len(cached)+1):
            score, _, tokens = runner.score(lookup[int(row.Row_Index)])
            rows.append({"Instance_ID": row.Instance_ID, "label": row.label, "Domain": row.Domain,
                         "Binoculars_Score": score, "Observed_Tokens": tokens})
            if number % 1000 == 0:
                pd.DataFrame(rows).to_csv(partial, mode="a" if partial.exists() else "w", header=not partial.exists(), index=False)
                rows = []
                print(f"  Binoculars {name}: {number}/{len(frame)}", flush=True)
        saved = pd.read_csv(partial) if partial.exists() else pd.DataFrame()
        result = pd.concat([saved, pd.DataFrame(rows)], ignore_index=True)
        temporary = path.with_name(path.stem+".tmp.gz")
        result.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(path)
        return result
    selection = predict("valid_select")
    threshold = select_score_threshold(selection.label.to_numpy(), selection.Binoculars_Score.to_numpy())
    calibration = predict("valid_calib")
    margin = calibration.Binoculars_Score.to_numpy()-threshold
    mapping_path = folder / "binoculars_mapping.json"
    if args.resume and mapping_path.exists():
        mapping = json.loads(mapping_path.read_text())
        threshold, scale, fitting_sec, success = [mapping[k] for k in ["threshold", "positive_scale", "fit_elapsed_sec", "success"]]
    else:
        start = time.perf_counter()
        fitted = analysis.minimize(lambda theta: analysis.nll_value(calibration.label.to_numpy(), margin*np.exp(theta[0])),
            [0.], method="L-BFGS-B", bounds=[(-8., 12.)])
        scale, success = float(np.exp(fitted.x[0])), bool(fitted.success)
        fitting_sec = time.perf_counter()-start
        dump_json(mapping_path, {"threshold": threshold, "positive_scale": scale,
            "threshold_source": "valid_select: Macro-F1", "scale_source": "valid_calib: NLL",
            "success": success, "fit_elapsed_sec": fitting_sec,
            "score_definition": "performer shifted NLL divided by unshifted observer-to-performer cross entropy"})
    records = {key: [] for key in ["Classification", "Domain", "ROC_PR", "Timing", "Cost", "Baseline_Fits", "Predictions"]}
    for condition in ["Standard", "OOD GPT", "Paraphrase"] + (["External"] if "External" in frames else []):
        frame = predict(condition)
        frame["Logit"] = scale*(frame.Binoculars_Score-threshold)
        frame["Inference_Run_ID"] = "Binoculars_fixed"
        path = folder / (condition.replace(" ", "_")+"_predictions.csv.gz")
        frame.to_csv(path, index=False, compression="gzip")
        records["Predictions"].append({"Seed": 0, "Method": "Binoculars", "File": str(path),
            "SHA256": analysis.sha256(path), "Instances": len(frame)})
        for seed in args.seeds:
            result = performance_records(frame, seed, "Binoculars", condition)
            records["Classification"].append({**result[0], "Inference_Run_ID": "Binoculars_fixed"})
            records["Domain"].extend(result[3])
            fpr, tpr, _ = analysis.roc_curve(1-frame.label.to_numpy(), -frame.Logit.to_numpy())
            precision, recall, _ = analysis.precision_recall_curve(1-frame.label.to_numpy(), -frame.Logit.to_numpy())
            grid = np.linspace(0, 1, 201)
            records["ROC_PR"].extend({"Seed": seed, "Method": "Binoculars", "Condition": condition,
                "Grid": float(x), "ROC_TPR": float(t), "PR_Precision": float(p)}
                for x, t, p in zip(grid, np.interp(grid, fpr, tpr), np.interp(grid, recall[::-1], precision[::-1])))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    text_rows = timing_texts(args, frames["Standard"], 2026)
    for i in range(args.timing_warmup+args.timing_runs):
        instance_id, text = text_rows[i % len(text_rows)]
        sync(device)
        start = time.perf_counter()
        score, model_ms, tokens = runner.score(text, measure=True)
        probability = float(analysis.expit(scale*(score-threshold)))
        sync(device)
        wall = 1000*(time.perf_counter()-start)
        if i >= args.timing_warmup:
            records["Timing"].append({"Seed": 0, "Method": "Binoculars", "Kmax": 1,
                "Timing_Run": i-args.timing_warmup, "Timing_ID": f"Binoculars_fixed:{i-args.timing_warmup}",
                "Instance_ID": instance_id, "End_to_End_ms": wall, "CUDA_Model_ms": model_ms,
                "Selected_Token_Chunks": 1, "Human_Probability": probability})
    counts = sum(model.num_parameters() for model in runner.models)
    resources = {"Seed": 0, "Method": "Binoculars", "Independent_Training_Run_Count": 0,
        "Training_Wall_Sec": 0., "Additional_Stage_I_Wall_Sec": 0., "Training_Cost_Parent": "None",
        "Total_Params_M": counts/1e6, "Calibration_Fit_CPU_Sec": fitting_sec,
        "Training_Peak_Allocated_VRAM_GB": None, "Training_Peak_Reserved_VRAM_GB": None,
        "Training_GPU_Hours": 0., "Execution_Device": str(device),
        "Precision_Executed": "NF4 + CPU offload" if device.type == "cuda" else "fp32",
        "Hardware_CPU": config["hardware"]["CPU"], "Hardware_GPU": config["hardware"]["GPU"],
        "System_RAM_GB": config["hardware"]["System_RAM_GB"], "GPU_VRAM_GB": config["hardware"]["GPU_VRAM_GB"],
        "Batch_Size": 1, "Accumulation": 0, "Timing_Warmup": args.timing_warmup,
        "Timing_Runs": args.timing_runs, "Calibration_Param_Count": 1, **peak(device)}
    records["Cost"].append(resources)
    records["Baseline_Fits"].append({"Method": "Binoculars", "Fit_CPU_Sec": fitting_sec,
        "Objective": "NLL; positive scale; fixed development threshold", "Success": success,
        "Score_Threshold": threshold, "Positive_Scale": scale, "N_Calibration": len(calibration)})
    paths = {}
    for name, rows in records.items():
        if rows:
            path = folder / f"records_{name}.csv.gz"
            pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
            paths[name] = {"path": str(path.relative_to(out)), "sha256": analysis.sha256(path)}
    mapping = folder / "binoculars_mapping.json"
    dump_json(complete, {"identity": identity, "Seed": 0, "Method": "Binoculars",
        "Independent_Training_Run_Count": 0, "Epochs": 0, "Checkpoint": str(mapping.relative_to(out)),
        "Checkpoint_SHA256": analysis.sha256(mapping), "records": paths})
    del runner
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path(r"C:\Users\LEGION\Train AI or Not\MAGE"))
    p.add_argument("--output-dir", type=Path, default=Path("RIGAT_Actual_Run"))
    p.add_argument("--backbone", default="microsoft/deberta-v3-base")
    p.add_argument("--revision", default="main")
    p.add_argument("--longformer-backbone", default="allenai/longformer-base-4096")
    p.add_argument("--longformer-revision", default="main")
    p.add_argument("--longformer-length", type=int, default=4096)
    p.add_argument("--with-binoculars", action="store_true")
    p.add_argument("--binoculars-observer", default="tiiuae/falcon-7b")
    p.add_argument("--binoculars-performer", default="tiiuae/falcon-7b-instruct")
    p.add_argument("--binoculars-revision", default="main")
    p.add_argument("--binoculars-max-tokens", type=int, default=512)
    p.add_argument("--binoculars-gpu-budget-gb", type=float, default=2.5)
    p.add_argument("--binoculars-cpu-budget-gb", type=float, default=10.)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--methods", nargs="+", choices=AVAILABLE_METHODS, default=METHODS)
    p.add_argument("--full-suite", action="store_true", help="All trainable baselines plus frozen Binoculars")
    p.add_argument("--run-only", nargs="+", choices=AVAILABLE_METHODS+["Binoculars"], help="Execute a subset of the locked methods; can resume others later")
    p.add_argument("--seed-only", nargs="+", type=int, help="Execute a subset of the locked seeds; can resume others later")
    p.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027, 2028, 2029, 2030])
    p.add_argument("--split-seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--accumulation", type=int, default=16)
    p.add_argument("--chunk-length", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--max-chunks", type=int, default=8)
    p.add_argument("--chunk-batch", type=int, default=1)
    p.add_argument("--encoder-lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--warmup-ratio", type=float, default=.1)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=.1)
    p.add_argument("--dp-noise-scales", nargs="+", type=float, default=[0., .01, .03, .06, .1])
    p.add_argument("--dp-alignment", type=float, default=.01)
    p.add_argument("--dp-policy-weight", type=float, default=.01)
    p.add_argument("--dp-entropy", type=float, default=.01)
    p.add_argument("--wave-scales", nargs="+", type=float, default=[1., 2., 4., 8., 16.])
    p.add_argument("--wave-kernel", type=int, default=31)
    p.add_argument("--wave-bins", type=int, default=32)
    p.add_argument("--lambda-local", type=float, default=.25)
    p.add_argument("--lambda-invariant", type=float, default=.10)
    p.add_argument("--lambda-robust", type=float, default=.50)
    p.add_argument("--group-eta", type=float, default=.05)
    p.add_argument("--reference-docs", type=int, default=12000)
    p.add_argument("--distance-normalization", choices=["none", "train-median"], default="none")
    p.add_argument("--trust-l2", type=float, default=1e-4)
    p.add_argument("--calibration-maxiter", type=int, default=600)
    p.add_argument("--timing-warmup", type=int, default=50)
    p.add_argument("--timing-runs", type=int, default=1000)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--skip-coverage", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--keep-optimizer-checkpoints", action="store_true")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--external-csv", type=Path)
    p.add_argument("--external-protocol", type=Path)
    p.add_argument("--max-train-docs", type=int, default=0)
    p.add_argument("--max-valid-docs", type=int, default=0)
    p.add_argument("--max-test-docs", type=int, default=0)
    args = p.parse_args(analysis.runtime_arguments(argv))
    if args.full_suite:
        args.methods = list(AVAILABLE_METHODS)
        args.with_binoculars = True
    for name in ["epochs", "accumulation", "chunk_length", "max_chunks", "chunk_batch", "reference_docs",
                 "calibration_maxiter", "timing_runs", "log_every", "cpu_threads"]:
        if getattr(args, name) < 1:
            p.error(f"{name} must be positive")
    if len(args.seeds) != len(set(args.seeds)) or len(args.methods) != len(set(args.methods)):
        p.error("Seeds and methods must be distinct")
    if args.timing_warmup < 0 or args.checkpoint_every < 0 or not 0 <= args.warmup_ratio <= 1:
        p.error("Invalid warmup/checkpoint settings")
    if not 2 <= args.overlap < args.chunk_length or not 0 <= args.dropout < 1 or args.longformer_length < 3:
        p.error("Invalid chunk overlap/dropout")
    for name in ["encoder_lr", "head_lr", "clip"]:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            p.error(f"{name} must be finite and positive")
    for name in ["trust_l2", "lambda_local", "lambda_invariant", "lambda_robust", "group_eta", "weight_decay"]:
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            p.error(f"{name} must be finite and nonnegative")
    if args.run_only and not set(args.run_only) <= set(args.methods+(["Binoculars"] if args.with_binoculars else [])):
        p.error("run-only must be a subset of methods")
    if args.seed_only and not set(args.seed_only) <= set(args.seeds):
        p.error("seed-only must be a subset of seeds")
    if min(args.seeds) < 0:
        p.error("Seeds must be nonnegative")
    if args.wave_kernel < 3 or args.wave_kernel % 2 != 1 or args.wave_bins < 1:
        p.error("Wave kernel must be odd and >=3; wave bins must be positive")
    if any(not math.isfinite(x) or x <= 0 for x in args.wave_scales) or any(not math.isfinite(x) or x < 0 for x in args.dp_noise_scales):
        p.error("Invalid wave scales/noise scales")
    if args.binoculars_max_tokens < 2 or min(args.binoculars_gpu_budget_gb, args.binoculars_cpu_budget_gb) <= 0:
        p.error("Invalid Binoculars context or memory budget")
    if bool(args.external_csv) != bool(args.external_protocol):
        p.error("external-csv and external-protocol must be supplied together")
    if any(x < 0 for x in [args.max_train_docs, args.max_valid_docs, args.max_test_docs]):
        p.error("Document limits must be nonnegative (0 means all documents)")
    if any(not math.isfinite(x) or x < 0 for x in [args.dp_alignment, args.dp_policy_weight, args.dp_entropy]):
        p.error("DP loss coefficients must be finite and nonnegative")
    for name in ["backbone", "longformer_backbone", "binoculars_observer", "binoculars_performer"]:
        value = Path(getattr(args, name))
        if value.is_dir():
            setattr(args, name, str(value.resolve()))
    args.coverage_k = sorted(set([k for k in [1, 2, 4, 6, 8] if k <= args.max_chunks] + [args.max_chunks]))
    return args


def main(argv=None):
    args = parse_arguments(argv)
    torch.set_num_threads(args.cpu_threads)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu or install a compatible PyTorch build")
    if args.with_binoculars and device.type == "cuda":
        if importlib.util.find_spec("bitsandbytes") is None or importlib.util.find_spec("accelerate") is None:
            raise RuntimeError("Install requirements_binoculars_gpu.txt before the full suite; Binoculars GPU offload dependencies are missing")
    frames, manifests, domain_ids = load_partitions(args, out, write=False)
    backbone_config = AutoConfig.from_pretrained(args.backbone, revision=args.revision, local_files_only=args.local_files_only)
    resolved_revision = getattr(backbone_config, "_commit_hash", None) or args.revision
    longformer_config = None
    longformer_revision = args.longformer_revision
    if "Longformer" in args.methods:
        longformer_config = AutoConfig.from_pretrained(args.longformer_backbone,
            revision=args.longformer_revision, local_files_only=args.local_files_only)
        longformer_revision = getattr(longformer_config, "_commit_hash", None) or args.longformer_revision
        if longformer_config.model_type != "longformer":
            raise ValueError("longformer-backbone must have model_type=longformer")
    binoculars_references = []
    if args.with_binoculars:
        for name in [args.binoculars_observer, args.binoculars_performer]:
            metadata = AutoConfig.from_pretrained(name, revision=args.binoculars_revision, local_files_only=args.local_files_only)
            binoculars_references.append({"model": name, "revision": getattr(metadata, "_commit_hash", None) or args.binoculars_revision})
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in {"resume", "preflight", "output_dir", "run_only", "seed_only"}}
    config.update({"version": VERSION, "device_resolved": str(device), "hardware": hardware(device),
                   "domain_ids": domain_ids, "generator_families": GENERATOR_FAMILIES,
                   "packages": {n: importlib.metadata.version(n) for n in ["torch", "transformers", "numpy", "pandas", "scipy", "scikit-learn", "openpyxl", "matplotlib", "tokenizers", "safetensors", "huggingface_hub"]},
                   "cuda_build": torch.version.cuda,
                   "cudnn_version": torch.backends.cudnn.version(),
                   "input_manifest": manifests,
                   "resolved_backbone_revision": resolved_revision,
                   "resolved_longformer_revision": longformer_revision if longformer_config else None,
                   "longformer_config": longformer_config.to_dict() if longformer_config else None,
                   "binoculars_references": binoculars_references,
                   "source_sha256": {name: analysis.sha256(Path(__file__).with_name(name)) for name in ["rigat_train.py", "rigat_generate_actual.py"]},
                   "local_pretrained_manifests": {name: local_model_manifest(name) for name in
                        [args.backbone]+([args.longformer_backbone] if longformer_config else [])+
                        ([args.binoculars_observer, args.binoculars_performer] if args.with_binoculars else [])},
                   "external_provenance": {"csv_sha256": analysis.sha256(args.external_csv), "protocol_sha256": analysis.sha256(args.external_protocol)} if args.external_csv else None,
                   "backbone_config": backbone_config.to_dict(),
                   "chunk_convention": {"encoder_sequence_length": args.chunk_length, "special_tokens": 2,
                        "content_payload": args.chunk_length-2, "stride": args.chunk_length-args.overlap,
                        "content_overlap": args.overlap-2, "position_embeddings": "selected-chunk rank",
                        "selection": "Rounded uniform indices including first/last; K=1 uses first"},
                   "partition_composition": [{"Split": name, "Instances": len(frame), "Human": int(frame.label.sum()),
                        "Machine": int((frame.label == 0).sum()), "Domains": frame.Domain.nunique()} for name, frame in frames.items()],
                   "mode": "subset_development" if any([args.max_train_docs, args.max_valid_docs, args.max_test_docs]) else "full_dataset",
                   "counts": {k: len(v) for k, v in frames.items()}})
    # HF label maps can use integer keys; normalize before comparing saved JSON.
    config = json.loads(json.dumps(analysis.json_safe(config), allow_nan=False))
    if args.with_binoculars and device.type == "cpu" and all("falcon-7b" in name.lower() for name in [args.binoculars_observer, args.binoculars_performer]):
        ram = config["hardware"]["System_RAM_GB"]
        if ram is not None and ram < 55:
            raise RuntimeError("The two Falcon-7B models in CPU fp32 exceed this host's RAM; use CUDA NF4/offload or a larger-memory host")
    print(f"Device: {device}; independent Stage I runs requested: {len(args.seeds)*len(args.methods)}", flush=True)
    print("Partitions:", config["counts"], flush=True)
    if args.preflight:
        dump_json(out / "preflight.json", config)
        print("Preflight complete; no weights loaded and no training started.", flush=True)
        return out
    previous = out / "experiment_config.json"
    if previous.exists() and json.loads(previous.read_text()) != config:
        raise ValueError("Output directory belongs to a different configuration; use a new output directory")
    if not previous.exists() and (any((out / "runs").glob("*/resume.pt")) or any((out / "runs").glob("*/run_complete.json"))):
        raise ValueError("Checkpoints exist without experiment_config.json; restore the original configuration")
    save_partition_metadata(frames, manifests, args, out)
    dump_json(out / "preflight.json", config)
    dump_json(previous, config)
    source_dir = out / "source"
    source_dir.mkdir(exist_ok=True)
    for name in ["rigat_train.py", "rigat_generate_actual.py"]:
        shutil.copy2(Path(__file__).with_name(name), source_dir / name)
    profiles = {}
    for profile in ["rigat"] + (["longformer"] if "Longformer" in args.methods else []):
        profile_args = copy.copy(args)
        revision = resolved_revision
        if profile == "longformer":
            profile_args.backbone = args.longformer_backbone
            profile_args.chunk_length, profile_args.overlap, profile_args.max_chunks = args.longformer_length, 2, 1
            revision = longformer_revision
        tokenizer = AutoTokenizer.from_pretrained(profile_args.backbone, revision=revision, local_files_only=args.local_files_only)
        tokenizer.save_pretrained(out / ("tokenizer" if profile == "rigat" else "tokenizer_longformer"))
        stores = {name: TokenStore(args.data_dir / filename, tokenizer, out / "token_cache") for name, filename in FILES.items()}
        if "External" in frames:
            stores["External"] = TokenStore(args.external_csv, tokenizer, out / "token_cache")
        profiles[profile] = (profile_args, revision, tokenizer, stores)
    word_values = []
    for block in pd.read_csv(args.data_dir / "test.csv", usecols=["text"], chunksize=2000):
        word_values.extend(block.text.astype(str).str.findall(r"\S+").str.len().tolist())
    standard_words = np.asarray(word_values, dtype=np.int64)[frames["Standard"].Row_Index.to_numpy()]
    for seed in args.seed_only or args.seeds:
        for method in [m for m in (args.run_only or args.methods) if m != "Binoculars"]:
            run_dir = out / "runs" / f"{seed}_{method}"
            run_dir.mkdir(parents=True, exist_ok=True)
            identity = hashlib.sha256(json.dumps({"config": config, "seed": seed, "method": method}, sort_keys=True).encode()).hexdigest()
            completion = run_dir / "run_complete.json"
            if completion.exists():
                if not args.resume:
                    raise FileExistsError(f"Completed run exists: {run_dir}; use --resume or a new directory")
                if json.loads(completion.read_text())["identity"] != identity:
                    raise ValueError("Completed-run identity mismatch")
                continue
            if (run_dir / "resume.pt").exists() and not args.resume:
                raise FileExistsError(f"Checkpoint exists: {run_dir}; use --resume or a new directory")
            seed_all(seed)
            run_args, run_revision, tokenizer, stores = profiles["longformer" if method == "Longformer" else "rigat"]
            backbone, loading_info = AutoModel.from_pretrained(run_args.backbone, revision=run_revision,
                local_files_only=args.local_files_only, output_loading_info=True)
            loading_report = {key: [str(x) for x in loading_info.get(key, [])] for key in
                              ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]}
            dump_json(run_dir / "pretrained_loading_info.json", loading_report)
            if loading_report["mismatched_keys"] or any(not key.startswith("pooler.") for key in loading_report["missing_keys"]):
                raise RuntimeError("Pretrained encoder weights are incomplete; inspect pretrained_loading_info.json")
            backbone.config.save_pretrained(run_dir / "encoder_config")
            model = RIGAT(backbone, method, len(domain_ids), run_args.max_chunks, args.dropout, not args.no_gradient_checkpointing,
                          args.dp_noise_scales, args.wave_scales, args.wave_kernel, args.wave_bins).float()
            print(f"Training {method}, seed {seed}...", flush=True)
            elapsed, train_peak, history = train_stage_one(model, frames["train"], frames["valid_select"],
                stores, tokenizer, run_args, device, run_dir, seed, identity)
            records, resources, predictions = run_evaluation(model, frames, stores, tokenizer, run_args, device, run_dir, seed, standard_words)
            count = sum(p.numel() for p in model.parameters())
            for row in resources:
                row.update({"Training_Wall_Sec": elapsed, "Training_Cost_Parent": method,
                    "Additional_Stage_I_Wall_Sec": elapsed if row["Method"] != "A5" else 0.,
                    "Total_Params_M": count/1e6, "Training_Peak_Allocated_VRAM_GB": train_peak["Peak_Allocated_VRAM_GB"],
                    "Training_Peak_Reserved_VRAM_GB": train_peak["Peak_Reserved_VRAM_GB"],
                    "Execution_Device": str(device), "Batch_Size": 1, "Accumulation": args.accumulation,
                    "Precision_Executed": args.precision if device.type == "cuda" else "fp32",
                    "Hardware_CPU": config["hardware"]["CPU"], "Hardware_GPU": config["hardware"]["GPU"],
                    "System_RAM_GB": config["hardware"]["System_RAM_GB"], "GPU_VRAM_GB": config["hardware"]["GPU_VRAM_GB"],
                    "Training_GPU_Hours": elapsed/3600 if device.type == "cuda" else np.nan,
                    "Calibration_Param_Count": 4 if row["Method"] == "A5" else 0,
                    "Timing_Warmup": args.timing_warmup, "Timing_Runs": args.timing_runs})
            records["Cost"] = resources
            records["History"] = history.to_dict("records")
            records["Predictions"] = predictions
            paths = {}
            for name, rows in records.items():
                if rows:
                    path = run_dir / f"records_{name}.csv.gz"
                    pd.DataFrame(rows).to_csv(path, index=False, compression="gzip")
                    paths[name] = {"path": str(path.relative_to(out)), "sha256": analysis.sha256(path)}
            dump_json(completion, {"identity": identity, "Seed": seed, "Method": method,
                "Independent_Training_Run_Count": 1, "Epochs": args.epochs,
                "Checkpoint": str((run_dir / "model.safetensors").relative_to(out)),
                "Checkpoint_SHA256": analysis.sha256(run_dir / "model.safetensors"), "records": paths})
            analysis.export_training_workbook(out)
            if not args.keep_optimizer_checkpoints:
                (run_dir / "resume.pt").unlink(missing_ok=True)
            del model, backbone
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    if args.with_binoculars and (not args.run_only or "Binoculars" in args.run_only):
        print("Evaluating frozen Binoculars once; no detector training or seed noise...", flush=True)
        run_binoculars(args, config, frames, device, out)
    analysis.export_training_workbook(out)
    status = json.loads((out / "completion_summary.json").read_text())
    print(f"Independent Stage I runs complete: {status['training_runs_completed']}/{status['training_runs_planned']}", flush=True)
    print(f"Completed measured-result workbook: {out / 'RIGAT_Training_Results.xlsx'}", flush=True)
    return out


if __name__ == "__main__":
    main()
