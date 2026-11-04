# IWDD — Vision-Language + Temporal Classification on MIVIA-IWDD-500

Reimplementation of the pipeline from:

> Kairanbay, M. *Vision-Language Temporal Analysis for Illegal Waste Dumping
> Detection in Surveillance Videos.* WACV 2026 Workshops, pp. 559–566.

Run on the MIVIA-IWDD-500 dataset (IWDD Contest 2026), plus 3D CNN baselines
the paper does not have.

```
frames @5 FPS → VLM (frozen, 5 prompts) → captions → Sentence-BERT 384-d
              → CNN+GRU → per-frame scores → smoothing + dual threshold → onset
```

**The VLM is never trained.** It is an inference-only feature extractor, which
is why the pipeline needs no caption ground truth — the only supervision used
anywhere is the video-level label and the onset timestamp that the dataset
actually ships.

---

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

`bitsandbytes` is only needed for 8-bit/4-bit backends (`cosmos_nano`, or
Qwen3-VL on a small GPU).

## Run

```bash
# 0. Inspect the data and see which annotations are broken
python scripts/01_inspect_dataset.py --dataset-root DATASET

# 1. Extract features. Expensive, runs once per VLM, then cached.
python scripts/02_extract_features.py --backend dummy --limit 4   # plumbing check
python scripts/02_extract_features.py --backend smolvlm           # the real thing

# 2. Train one architecture on the full training split
python scripts/03_train.py --arch cnn_gru \
  --signature smolvlm-SmolVLM-500M-Instruct-multi-full-fps5-concat_text

# 3. Or 5-fold cross-validation on a 50-video subset (the paper protocol)
python scripts/03_train.py --arch cnn_gru --cv --folds 5 --subset 50 --signature ...

# 4. Compare all six architectures against one cached feature set
python scripts/04_benchmark.py --signature ...

# 5. Pixel-space control, no VLM involved
python scripts/03_train.py --arch conv3d --clip-len 16
```

## Layout

| Module | Responsibility |
|---|---|
| `iwdd/config.py` | Typed dataclass config for every stage; YAML-loadable |
| `iwdd/annotations.py` | JSON parsing, schema validation, split statistics |
| `iwdd/sampling.py` | 5 FPS frame extraction with fractional stride |
| `iwdd/labeling.py` | Onset timestamp → per-frame binary labels |
| `iwdd/captioning/` | Prompts + VLM backends (SmolVLM, Qwen3-VL, Cosmos, BLIP, dummy) |
| `iwdd/encoders.py` | Captions → vectors (`concat_text`, `per_aspect`, `vision`) |
| `iwdd/features.py` | Orchestration + the `.npz` feature cache |
| `iwdd/datasets.py` | Padded sequence batches, clip windows, stratified folds |
| `iwdd/models/` | LSTM, GRU, CNN+LSTM, CNN+GRU, BiLSTM+Attn, Transformer, Conv3D, R3D-18 |
| `iwdd/postprocess.py` | Moving average + dual-threshold event trigger |
| `iwdd/metrics.py` | IWDD Contest 2026 scoring |
| `iwdd/training.py` | Trainer, CrossValidator, BenchmarkRunner |
| `iwdd/clip_training.py` | 3D CNN training and clip→timeline score reassembly |

## Configuration

`configs/default.yaml` reproduces the paper. Key settings and their source:

| Setting | Value | Source |
|---|---|---|
| Sampling rate | 5 FPS | §3.2.1 |
| Prompts | 5 focused aspects | §3.2.2, verbatim |
| Sentence encoder | all-MiniLM-L6-v2, 384-d | Eq. 6 |
| Optimiser | Adam 1e-3, ReduceLROnPlateau(0.5, 3), clip 1.0 | §3.3, §4.2 |
| Batch / epochs / hidden / dropout | 4 / 20 / 256 / 0.3 | §4.2 |
| Positive class weight | `w+ = 8.0` | §3.3 — **see caveat below** |
| Smoothing / thresholds | w=9, peak 0.7, run 0.6, ≥5 frames | §3.2.4 |
| Detection tolerance | `g − 3.0s ≤ p ≤ g + 10.0s` | §3.4 |
| Positive window length | 2.0s | **not in the paper** — see below |

---

## What this dataset actually contains

Verified against all 500 files on disk, not the dataset README:

- **400 train / 100 test**, exactly 50/50 positive/negative in both splits.
- Annotation fields: `label` (the **string** `"true"`/`"false"`), `timestamp`
  (integer seconds, `-1` for negatives), and metadata with `video_id`,
  `resolution`, `frame_rate`, `duration`, `event_type`, `time_of_day`,
  `lighting`.
- The dataset README omits `time_of_day` and `lighting` from its schema
  example. **They are present on disk** (`day`/`night`, `high`/`low`).
- Durations span **2.9s to 109.5s** (median 18.6s) and frame rates span
  **6.0 to 59.94 FPS** across 20 distinct values in the training split. At
  5 FPS that is 14 to 548 frames per clip, so sequences are padded and masked
  everywhere.
- 31 distinct resolutions in training, from 320×240 to 4096×2160.
- All onsets are **whole integers** — no sub-second precision exists.

### Broken annotations

`scripts/01_inspect_dataset.py` reports these; `drop_invalid: true` removes the
fatal one.

| Video | Problem |
|---|---|
| `vid0481` | onset 17.0s but duration 14.06s — **dropped** |
| `vid0020`, `vid0111`, `vid0139`, `vid0156`, `vid0185`, `vid0212`, `vid0268` | under 1s remaining after onset (train) |
| `vid0127` | under 1s remaining after onset (test) |

---

## Deviations from the paper, and why

These are deliberate. Each one is a place where the paper is silent,
self-inconsistent, or wrong about this corpus.

**1. The positive window is invented.** The dataset gives an onset but no event
end time, so per-frame labels have to be synthesised. The default of 2.0s after
onset follows §6.1 ("a local temporal event (5–15 frames)"). Nothing verifies
it. Sweep it — `relabel_cache()` rewrites labels without re-captioning.

**2. `w+ = 8.0` does not match a 2.0s window on this corpus.** Measured over
the real training split:

| `post_seconds` | positive frame ratio | implied `w+` |
|---|---|---|
| 1.0 | 2.52% | 38.7 |
| 2.0 | 4.48% | **21.3** |
| 3.0 | 6.15% | 15.3 |
| 5.0 | 8.87% | 10.3 |
| 6.0 | 9.93% | 9.1 |

`w+ = 8` corresponds to a window of roughly 7 seconds, which contradicts the
paper's own "5–15 frames". Both paper values are kept as defaults for fidelity;
pass `--auto-positive-weight` to derive `w+` from the data instead.

**3. Loss is masked and averaged across the batch.** The paper does not mention
padding. With clips from 2.9s to 109.5s, averaging per sequence would let a
109s video count 38× a 2.9s one.

**4. A positive video detected at the wrong time counts as both FP and FN.**
That is the literal reading of §3.4, and it is the right incentive — an alarm
at the wrong moment is a false alarm *and* a miss.

**5. Recurrent depth is 2, hidden width 256 per direction.** The paper reports
parameter counts but not the layer configuration. This setting reproduces two
of them exactly:

| Architecture | This repo | Paper | Δ |
|---|---|---|---|
| `lstm` | 2,893,313 | 2,893,313 | **0** |
| `gru` | 2,170,369 | 2,170,369 | **0** |
| `cnn_lstm` | 2,926,849 | 2,878,081 | +48,768 |
| `cnn_gru` | 2,269,441 | 2,220,673 | +48,768 |

(The exact match required a `LayerNorm` before the output head — a bare
`Linear` is 1024 parameters short in both cases.) The hybrids are 2% over
because the paper never reports `conv_channels` or `conv_kernel_size`; the
constant offset means one unreported conv setting explains both.

**6. Smoothing reflects at the edges** rather than zero-padding. One training
video has its onset at t=0, which a zero-padded moving average would suppress.

---

## Additions beyond the paper

- **3D CNN controls** (`conv3d` from scratch, `r3d18` pretrained on Kinetics).
  Every architecture in the paper sees the video only through generated
  captions, so none of them can tell you whether the vision-language detour
  earns its 1050ms per frame. These answer that directly. Clip scores are
  scattered back onto the time axis and fed through the identical detector, so
  the comparison is apples to apples.
- **`vision` encoder mode** — pools the VLM vision tower and skips text
  generation entirely. Captioning is ~90% of runtime in the paper, so this is
  the only credible route to real-time edge inference.
- **`per_aspect` encoder mode** — encodes each of the five captions separately
  (5×384) instead of averaging them into one sentence vector.
- **`tune_thresholds()`** — the 0.7/0.6 thresholds were tuned for BLIP. A
  different VLM will want different ones. Tune on training folds only.

## Cost planning

400 videos × ~23s × 5 FPS ≈ **46,700 frames**, × 5 prompts = **233,000
generations**. At the paper's 0.85 FPS that is ~15 hours for the training
split alone.

The cache is what makes this tractable: extract once, then every architecture
and every hyper-parameter sweep reads `.npz` files and finishes in minutes.
Cache directories are keyed by backend, model, prompt mode, quantisation,
sampling rate and encoder mode, so different VLMs never collide.

## Status

Verified end to end on real data with the `dummy` backend (no GPU): annotation
parsing, video decoding, 5 FPS sampling, onset→frame labelling, caching, padded
batching, all six sequence architectures, 5-fold stratified CV, post-processing
and contest scoring all run. Accuracy numbers from the dummy backend are
meaningless by construction — it emits colour histograms, not semantics.

**Not yet run:** a real VLM backend (SmolVLM / Qwen3-VL / Cosmos / BLIP) or
Sentence-BERT, because this machine has no GPU and no `transformers` install.
Those code paths are implemented but unexercised — expect to debug model-specific
processor quirks on first contact, particularly chat-template handling and
`Qwen3VLForConditionalGeneration` availability in your `transformers` version.
Verify `caption.model_id` against the model card before a multi-hour run.
