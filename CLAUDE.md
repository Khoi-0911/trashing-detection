# CLAUDE.md — working notes for this repo

Context for anyone (human or agent) picking this up. The [README](README.md)
describes *what the code does*; this file records *what was learned building
it* — the traps, the verified facts, and the mistakes already made so they are
not repeated.

---

## 1. What exists

A reimplementation of Kairanbay, *Vision-Language Temporal Analysis for Illegal
Waste Dumping Detection in Surveillance Videos* (WACV 2026 Workshops, 559–566),
run against MIVIA-IWDD-500 / IWDD Contest 2026.

```
frames @5 FPS → frozen VLM, 5 prompts → captions → SBERT 384-d
              → CNN+GRU → per-frame scores → smooth + dual threshold → onset
```

Package `iwdd/` (~6,400 LOC, 24 files), CLI in `scripts/`, config in
`configs/default.yaml`. Source paper PDF: `../ref_docs/BLIP_CNN_GRU.pdf`.

**The single most important structural fact:** the VLM is never trained. It is
an inference-only feature extractor. That is why this pipeline works on a
dataset that has *no caption ground truth* — the only supervision anywhere is
`label` + `timestamp`. Anyone proposing to "fine-tune the VLM on this dataset"
should re-read this paragraph first; it is a different project.

---

## 2. Environment (this machine)

| Thing | Reality |
|---|---|
| `python` on PATH | `C:\msys64\mingw64\bin\python.exe` — **no pip, no packages** |
| Conda base | `C:\Users\nhok0\miniconda3\python.exe` — Python 3.14, has pip, no packages |
| Working venv | `.venv/` in the project root — **use this** |
| torch | 2.15.0+**cpu** — no GPU on this machine |
| Installed in venv | numpy, PyYAML, opencv-python, Pillow, torch |
| **Not** installed | transformers, sentence-transformers, torchvision, bitsandbytes |

Always invoke `.\.venv\Scripts\python.exe` explicitly. Plain `python` hits the
msys2 build and fails on `import numpy`.

---

## 3. Dataset facts — verified against all 500 files, not the dataset README

```
DATASET/TRAINING_SET/{videos,labels}     400 mp4 + 400 json
DATASET/VALIDATION_SET/{videos,labels}   100 mp4 + 100 json
```

- Exactly 50/50 positive/negative in both splits (200/200 and 50/50).
- Video ids are **global and interleaved**: `vid0005` and `vid0012` are in
  VALIDATION_SET while `vid0001`–`vid0004` are in TRAINING_SET. Never assume
  contiguous numbering per split.
- `label` is the **string** `"true"`/`"false"`, not a JSON boolean.
  `if payload["label"]:` is always True. This trap is why
  `VideoAnnotation.from_json` parses strictly and raises on anything else.
- `timestamp` is **integer seconds**, `-1` for negatives. All 250 onsets are
  whole integers — there is no sub-second precision in this corpus.
- The real schema **has `time_of_day` and `lighting`**, which the dataset
  README's example JSON omits. Values are `day`/`night` and `high`/`low` (not
  `BRIGHT`/`DIM` as the README table claims).
- Durations: **2.9s to 109.5s**, median 18.6s. At 5 FPS that is 14 to 548
  frames. Everything downstream must pad and mask.
- Frame rates: **6.0 to 59.94**, 20 distinct values in train. The sampling
  stride is fractional and must be accumulated in float or it drifts by
  seconds on long clips.
- 31 distinct resolutions in train, 320×240 to 4096×2160.

### Known-bad annotations

| Video | Problem | Handling |
|---|---|---|
| `vid0481` | onset 17.0s > duration 14.06s | dropped by `drop_invalid: true` |
| `vid0020` `vid0111` `vid0139` `vid0156` `vid0185` `vid0212` `vid0268` | <1s after onset (train) | warned, kept |
| `vid0127` | <1s after onset (test) | warned, kept |

`scripts/01_inspect_dataset.py` reports all of these. Run it first, always.

---

## 4. Where the paper is wrong, silent, or self-contradictory

Do not treat the paper as ground truth. Specific issues found:

1. **Confusion matrix does not reconcile with reported precision/recall.**
   §5.1.1 gives TP=20, TN=23, FP=2, FN=5 → precision 0.909, recall 0.800.
   Table 2 says 0.6511 / 0.9000. Accuracy 86% and specificity 92% do check
   out. Do not use 0.7429 as a hard comparison target.

2. **VLM count is inconsistent.** Abstract says five VLMs, the contributions
   bullet says four, §4.3 says 24 configurations (6×4), Table 1 has 5 rows
   (=30). §4.2 lists model ids for only 4.

3. **`w+ = 8.0` contradicts the stated window.** §6.1 calls dumping a "local
   temporal event (5–15 frames)" = 1–3s at 5 FPS, but measured on the real
   training split:

   | `post_seconds` | positive frame ratio | implied `w+` |
   |---|---|---|
   | 1.0 | 2.52% | 38.7 |
   | **2.0** | **4.48%** | **21.3** |
   | 3.0 | 6.15% | 15.3 |
   | 5.0 | 8.87% | 10.3 |
   | 6.0 | 9.93% | 9.1 |

   `w+ = 8` implies a ~7s window. Both paper values are kept as defaults for
   fidelity; `--auto-positive-weight` derives it from data instead.

4. **"Frame-level annotations" claim is false.** §6.4 says MIVIA-IWDD has
   "frame-level annotations". It does not — the dataset README explicitly says
   dense frame-level labels are unavailable. The paper means the onset
   timestamp. Per-frame labels must be synthesised; that is `labeling.py`.

5. **All results are on 50 of the 400 videos**, 5-fold CV with 10 videos per
   validation fold, F1 std ±0.191. The real confidence interval spans roughly
   0.55–0.93. Running the full 400 is the paper's own stated future work and
   the clearest contribution available here.

6. **0.85 FPS on an A100**, with captioning at 89.7% of runtime. Not real-time
   on any hardware, let alone edge.

7. **Layer configuration is unreported** but recoverable from the parameter
   counts — see §5 below.

---

## 5. Non-obvious implementation decisions

**Recurrent models: 2 layers, hidden 256 *per direction*, `LayerNorm` before
the head.** This was reverse-engineered from the paper's parameter counts and
reproduces two of them exactly:

| | This repo | Paper | Δ |
|---|---|---|---|
| `lstm` | 2,893,313 | 2,893,313 | **0** |
| `gru` | 2,170,369 | 2,170,369 | **0** |
| `cnn_lstm` | 2,926,849 | 2,878,081 | +48,768 |
| `cnn_gru` | 2,269,441 | 2,220,673 | +48,768 |

The `LayerNorm` is load-bearing: a bare `Linear` head is exactly 1024
parameters short in both exact cases, which is `LayerNorm(512)`. The hybrids
are over by a constant +48,768, meaning one unreported conv setting
(`conv_channels` / `conv_kernel_size`) explains both. Do not "fix" the hybrids
by fudging conv sizes without evidence.

**Loss is masked and averaged over valid frames across the whole batch**, not
per sequence. With 2.9s–109.5s clips, per-sequence averaging would let a 109s
video count 38× a 2.9s one.

**A positive video detected at the wrong time counts as both FP and FN.** That
is the literal reading of §3.4 and the right incentive. Implemented in
`IWDDEvaluator._score_one`, which returns the outcome string `"FP+FN"`.

**Smoothing reflects at the edges** instead of zero-padding, because one
training video has its onset at t=0 and zero-padding would suppress it.

**Feature cache is keyed by a signature** (`backend-model-prompts-quant-fps-mode`)
so different VLMs never collide. Extract once, then every architecture and
sweep reads `.npz` in minutes. Without this, each sweep costs another ~15h.
`relabel_cache()` rewrites frame labels without re-captioning — use it to sweep
`post_seconds`.

---

## 6. Mistakes already made — do not repeat

**Bash heredocs break on apostrophes in this environment.** Writing
`iwdd/config.py` via `cat > file <<'PY'` failed with *"unexpected EOF while
looking for matching `'`"* because a docstring contained `the paper's`. Quoted
heredocs should make this safe; on this setup they do not.
→ **Use the Write tool for code files.** Reserve heredocs for short, 
apostrophe-free snippets.

**`cd` inside a Bash call persists and changes the session working directory.**
An early `cd DATASET` silently moved the primary working directory.
→ Use absolute paths, or `cd` as part of a compound command only.

**Do not estimate a distribution that can be measured.** The positive-frame
ratio was reasoned out as "~11%, matching w+=8". The measured value is 4.48%
and `w+` is 21.3. The reasoning forgot that negative videos contribute zero
positive frames, halving the ratio. One script run settled it.
→ Measure first, then reason about the number.

**Do not declare something unavailable before checking what the user already
has.** The IWDD Contest metric definitions were reported as "unknown, site
unreachable" — they were in §3.4 of a paper sitting in `../ref_docs/`.

**`ClipDataset` originally re-decoded the entire video for every window** —
707 full decodes for 100 videos, turning a 4-minute eval into >10 minutes.
Fixed with a bounded LRU keyed by `video_id`. If `ClipDataset` is ever changed,
keep the cache; sequential (non-shuffled) window order makes a 2-entry cache
sufficient.

**Typo `ayer := None` was introduced into a tuple literal** in
`HFChatCaptioner._find_vision_tower` and only caught on review.
→ Compile after every file: `python -m compileall -q iwdd scripts`.

**A fallback that falls back to the thing that just failed.**
`HFChatCaptioner.load()` did `getattr(transformers, name, None)` and, on
`None`, assigned `transformers.AutoModelForVision2Seq` — the exact attribute
that had just been missing. It logged the self-refuting line *"transformers has
no AutoModelForVision2Seq, falling back to AutoModelForVision2Seq"* and then
raised `AttributeError` on every video. Root cause: **transformers 5.x removed
`AutoModelForVision2Seq`** in favour of `AutoModelForImageTextToText`, and
Colab ships 5.x.
→ Fixed by `_resolve_model_class()`, which walks
`[declared] + FALLBACK_MODEL_CLASSES` and raises a useful `ImportError` naming
the installed version if none exist. Verified against simulated 4.x and 5.x.
→ **Lesson: a fallback path must name a *different* target than the one that
failed, and must be exercised before shipping.** This one was never run,
because no real VLM backend had ever been loaded.

**NumPy scalars killed a 26-minute captioning run at the write step.**
`CaptionSet.timestamp` came from `list(numpy_array)`, so it held
`numpy.float32`, which `json.dumps` refuses. Captioning for all five videos
succeeded and was then discarded in `_write_captions`.
-> Fixed by coercing at the boundary (`float()` in `caption_frames`, explicit
`int`/`float`/`str` in `as_dict`) plus `default=str` on every `json.dumps`
that writes run output.
-> **Lesson: coerce NumPy scalars to built-ins at the serialization boundary,
and order the pipeline so cheap failures happen before expensive work** --
this crash was at the very end of the costly stage.

**Two crashes found by simulating a 5-video smoke test locally** (dummy
backend, before spending GPU time):

1. `stratified_folds()` produced **empty folds** when `num_folds` exceeded the
   rarest class count. 5 videos split 3 positive / 2 negative with
   `--folds 5` gave folds 4 and 5 zero validation videos, then
   `ValueError: SequenceDataset received no items`.
   -> Now capped at `min(class counts)` with a warning, and empty folds are
   dropped. 40 videos still give a clean 5 folds, so no regression.
2. **Holdout training died on an unextracted test split.** Extracting with
   `--splits TRAINING_SET` then training without `--cv` wants
   VALIDATION_SET too; it logged ~100 individual cache-miss warnings and then
   raised the same opaque `ValueError`.
   -> `load_sequence_dataset()` now checks up front and raises a message
   naming the split, the extraction command, and the `--cv` alternative. It
   also summarises partial caches in one line instead of one per video.

-> **Lesson: simulate the user's exact situation with the cheap backend before
they spend GPU hours.** Both bugs sat in code paths that only a small or
partial cache reaches.

---

## 7. What is verified and what is not

**Verified by actually running on the real dataset** (with `DummyCaptioner` +
`--encoder-mode vision`, which needs no transformers):

- annotation parsing and validation on all 500 files
- OpenCV decoding and 5 FPS sampling (`vid0001` 5.77s → 29 frames ✓)
- onset → frame-label expansion, `.npz` cache round-trip
- padded batching with mask; padded logits forced to −30
- all six sequence architectures forward and train
- 5-fold stratified CV (32/8 per fold), with a real TP at 2.5s delay
- `conv3d` end to end: 707 windows / 100 videos → TP=2 FP=12 FN=48 TN=44
- detector and contest matching at all three boundary cases

**Partially exercised on Colab T4 (2026-11-04):** `smolvlm` reached model
loading and surfaced the `AutoModelForVision2Seq` removal (see §6). Model
loading itself is still unconfirmed past that point.

**Never executed — expect first-contact bugs:**

- real captioning end to end: `smolvlm`, `smolvlm2`, `qwen3vl`, `cosmos_nano`, `blip`
- `SentenceBertEncoder` (the paper's actual `concat_text` path)
- `r3d18` (needs torchvision)
- 8-bit / 4-bit quantisation (needs bitsandbytes)

Likely failure points in that untested code: `apply_chat_template` output shape
per model, the `images=[[img]]` nesting convention, the `dtype=` kwarg name
(renamed from `torch_dtype` in transformers 4.56), and `_find_vision_tower` not
matching a given model's module layout.

Environment note: Colab currently runs Python 3.13 with transformers 5.x. Any
API assumed from 4.x should be checked against 5.x before use.

`CosmosNanoCaptioner.default_model_id` is `nvidia/Cosmos-Reason1-7B` and is
**unverified** — check the model card before starting a multi-hour run.

---

## 8. Commands

```bash
# always this interpreter
.\.venv\Scripts\python.exe

# 0. inspect data + list broken annotations      (run this first, always)
scripts/01_inspect_dataset.py --dataset-root DATASET

# 1. extract features — expensive, once per VLM, then cached
scripts/02_extract_features.py --backend dummy --encoder-mode vision --limit 6
scripts/02_extract_features.py --backend smolvlm

# 2. train / 3. cross-validate / 4. sweep architectures
scripts/03_train.py --arch cnn_gru --signature <sig>
scripts/03_train.py --arch cnn_gru --cv --folds 5 --subset 50 --signature <sig>
scripts/04_benchmark.py --signature <sig>

# 5. pixel-space control, no VLM
scripts/03_train.py --arch conv3d --clip-len 16

# sanity
python -m compileall -q iwdd scripts
```

Cache signature format: `{backend}-{model}-{multi|single}-{full|8bit|4bit}-fps{N}-{encoder_mode}`.
The dummy smoke-test cache is `dummy-default-multi-full-fps5-vision`.

---

## 9. Open questions worth answering next

1. Does `vision` mode (no text generation) beat or match `concat_text`? The
   paper never tests it and it is ~100× cheaper. If it holds up, real-time
   edge inference becomes plausible.
2. What is the right `post_seconds`? No ground truth exists; sweep it against
   contest F1 on training folds only.
3. Does the 3D CNN control beat the captioning pipeline? If it does, the
   paper's central premise is in trouble on this corpus.
4. Re-tune `peak_threshold` / `run_threshold` per VLM — 0.7/0.6 were tuned for
   BLIP. `tune_thresholds()` exists for this. Training folds only.
5. Dataset has **no license** and the maintainers list licensing as unresolved.
   Email the authors (`agreco@unisa.it`, `anricciardi@unisa.it`) before
   publishing a model, generated captions, or results.

---

## 10. Measured throughput on a Colab T4 (2026-11-04)

SmolVLM-500M, fp16, `--batch-size 8`, `--max-new-tokens 48`, 5 prompts, 5 FPS.
Five videos, 356 frames, 1545 s wall clock:

| video | seconds | frames | fps |
|---|---|---|---|
| vid0001 | 129 | 29 | 0.225 |
| vid0002 | 113 | 26 | 0.230 |
| vid0003 | 670 | 156 | 0.233 |
| vid0004 | 402 | 92 | 0.229 |
| vid0006 | 231 | 53 | 0.229 |
| **total** | **1545** | **356** | **0.230** |

Remarkably stable across clip lengths. Extrapolated to the full corpus
(46,665 train + 11,234 test = 57,899 frames):

| scope | hours at 0.230 fps |
|---|---|
| TRAINING_SET | **56** |
| VALIDATION_SET | 14 |
| both | **70** |

**The stock configuration is not viable on a T4** against a 12-hour Colab
session. Rough projections for the full corpus:

| change | hours |
|---|---|
| baseline | 70 |
| `--batch-size 32` (peak VRAM was only 3.5 / 16 GB) | ~35 |
| + `--max-new-tokens 24` | ~19 |
| + `--target-fps 2` (needs `min_consecutive: 2`) | ~8 |
| + SmolVLM-256M | ~5 |

Known structural inefficiency, not yet addressed: `caption_frames()` loops
prompts on the outside, so the vision encoder re-encodes the same frame five
times. Encoding once and reusing the prefix KV cache across the five prompts
would cut prefill cost, but it is a non-trivial refactor and decode likely
dominates at 48 tokens.
