# Hướng dẫn đọc code — giải thích từng file, từng đối tượng

Tài liệu này đi qua **24 file** trong `iwdd/` và `scripts/`, giải thích mỗi lớp
và mỗi hàm làm gì, kèm ví dụ ngắn chạy được.

Luồng dữ liệu xuyên suốt — bám vào nó để biết mỗi file nằm ở đâu:

```
JSON + MP4 ──► AnnotationStore ──► FrameSampler ──► BaseCaptioner ──► BaseFrameEncoder
                     │                   │                                   │
                     └── FrameLabeler ───┴───────────► FrameFeatures ◄────────┘
                                                            │
                                                      FeatureCache (.npz)
                                                            │
                                                     SequenceDataset
                                                            │
                                                    BaseSequenceModel
                                                            │
                                                      EventDetector
                                                            │
                                                      IWDDEvaluator
```

Mọi ví dụ đều giả định chạy từ thư mục gốc dự án bằng `.\.venv\Scripts\python.exe`.

---

# PHẦN A — Nền tảng

## `iwdd/utils.py` — tiện ích dùng chung

Không chứa logic nghiệp vụ. Chỉ là hạ tầng: log, seed, đo thời gian, đo bộ nhớ.

### `class Timer`
Context manager đo thời gian một khối lệnh. Dùng để tính throughput (FPS) —
một chỉ số mà contest **chấm điểm**, nên nó không phải debug mà là output.

```python
from iwdd.utils import Timer
with Timer("captioning") as t:
    ...                      # việc cần đo
print(t.elapsed)             # 12.63  (giây)
```

### `class ResourceProbe`
Gom thống kê throughput + bộ nhớ GPU đỉnh qua nhiều video.

| Thành phần | Ý nghĩa |
|---|---|
| `.add(frames, seconds)` | ghi nhận một video đã xử lý |
| `.refresh_peak_memory()` | hỏi CUDA allocator mức đỉnh |
| `.fps` / `.fps_std` | tốc độ tổng / độ lệch giữa các video |
| `.peak_memory_gb` | bộ nhớ đỉnh (GiB) |
| `.as_dict()` | xuất ra dict để ghi log |

```python
from iwdd.utils import ResourceProbe
p = ResourceProbe()
p.add(frames=29, seconds=1.2)
p.add(frames=156, seconds=4.1)
print(round(p.fps, 2))       # 34.91   = (29+156) / (1.2+4.1)
```

### Các hàm
| Hàm | Công dụng |
|---|---|
| `setup_logging(level, log_file)` | tạo logger `iwdd`, ghi ra stdout và file |
| `get_logger()` | lấy logger đã cấu hình, không cấu hình lại |
| `set_seed(seed)` | seed `random`, `numpy`, `torch` + bật cuDNN deterministic |
| `resolve_device("auto")` | trả `"cuda"` nếu có, ngược lại `"cpu"` |
| `human_bytes(n)` | `8900000 → "8.49 MB"` |

```python
from iwdd.utils import human_bytes, resolve_device
print(human_bytes(8_900_000))   # 8.49 MB
print(resolve_device("auto"))   # cpu   (máy này không có GPU)
```

---

## `iwdd/config.py` — cấu hình có kiểu

**10 dataclass**, mỗi cái phụ trách một giai đoạn. Lý do dùng dataclass thay vì
`dict`: gõ sai tên key sẽ báo lỗi ngay thay vì âm thầm bị bỏ qua.

| Lớp | Giữ cái gì | Vài trường quan trọng |
|---|---|---|
| `PathsConfig` | đường dẫn | `dataset_root`, `cache_dir`, `output_dir` |
| `SamplingConfig` | trích frame | `target_fps=5.0`, `resize_shorter_side=384` |
| `LabelingConfig` | onset → nhãn frame | `post_seconds=2.0`, `drop_invalid=True` |
| `CaptionConfig` | VLM | `backend`, `model_id`, `load_in_8bit`, `single_prompt` |
| `EncoderConfig` | caption → vector | `mode`, `model_id` (SBERT) |
| `ModelConfig` | kiến trúc | `name`, `hidden_dim=256`, `num_layers=2` |
| `TrainConfig` | tối ưu | `epochs=20`, `positive_weight=8.0`, `num_folds=5` |
| `PostProcessConfig` | kích hoạt cảnh báo | `peak_threshold=0.7`, `run_threshold=0.6` |
| `MetricConfig` | chấm điểm | `tolerance_before=3.0`, `tolerance_after=10.0` |
| `ExperimentConfig` | gom tất cả | 9 trường trên + `run_name` |

### `PathsConfig.split_dir(split)`
Ghép đường dẫn tuyệt đối tới một split.

```python
from iwdd.config import PathsConfig
p = PathsConfig(dataset_root="DATASET")
print(p.split_dir("TRAINING_SET"))
# D:\Trashing_detection\trashing-detection\DATASET\TRAINING_SET
```

### `ExperimentConfig.from_yaml(path)` / `.from_dict(payload)`
Nạp từ YAML. **Key lạ sẽ raise `KeyError`** — đây là chủ đích, để lỗi chính tả
trong file config lộ ra ngay.

```python
from iwdd.config import ExperimentConfig
c = ExperimentConfig.from_yaml("configs/default.yaml")
print(c.model.name, c.train.positive_weight)   # cnn_gru 8.0

ExperimentConfig.from_dict({"model": {"nmae": "gru"}})
# KeyError: "Unknown fields in section 'model': ['nmae']"
```

### `ExperimentConfig.feature_signature()`
Sinh khóa định danh cho cache đặc trưng. **Đây là thứ giữ cho các VLM khác nhau
không ghi đè lên nhau.**

```python
print(c.feature_signature())
# smolvlm-default-multi-full-fps5-concat_text
#  backend  model   prompt quant fps  encoder-mode
```

Đổi `backend` sang `qwen3vl` → signature đổi → cache mới, cache cũ vẫn còn.

### `ExperimentConfig.save(path)` / `.to_dict()`
Ghi config ra JSON cạnh kết quả, để sau này biết run đó dùng tham số gì.

---

# PHẦN B — Đọc và chuẩn bị dữ liệu

## `iwdd/annotations.py` — đọc và kiểm tra JSON

### `class DisposalModality(str, Enum)`
Ba giá trị của `event_type`. `.parse()` chuẩn hóa và **raise nếu lạ**, thay vì
im lặng cho qua.

```python
from iwdd.annotations import DisposalModality
print(DisposalModality.parse("static dumping"))   # DisposalModality.STATIC
DisposalModality.parse("vứt rác")                 # ValueError
```

### `class VideoAnnotation` (frozen dataclass)
Một bản ghi = một video. Gộp ground truth + metadata + đường dẫn file.

**`.from_json(label_path, video_dir, split)`** — parse một file JSON.
Đây là nơi xử lý hai cái bẫy lớn nhất của dataset:

```python
# BẪY 1: label là CHUỖI "true"/"false", không phải bool.
#         if payload["label"]:  →  luôn True vì "false" là chuỗi khác rỗng.
# BẪY 2: timestamp = -1 nghĩa là "không có sự kiện", không phải giây -1.
```

```python
from pathlib import Path
from iwdd.annotations import VideoAnnotation
a = VideoAnnotation.from_json(
    Path("DATASET/TRAINING_SET/labels/vid0001.json"),
    Path("DATASET/TRAINING_SET/videos"), "TRAINING_SET")
print(a.video_id, a.is_positive, a.onset, a.duration, a.modality.value)
# vid0001 True 4.0 5.77 static dumping
```

| Thuộc tính | Ví dụ |
|---|---|
| `.resolution` | `(1280, 720)` — đã tách từ chuỗi `"1280x720"` |
| `.tail_seconds` | `1.77` — số giây còn lại sau onset; `None` nếu âm tính |

**`.validate(min_tail_seconds)`** — trả về danh sách vấn đề, rỗng nghĩa là sạch.

```python
bad = VideoAnnotation.from_json(
    Path("DATASET/TRAINING_SET/labels/vid0481.json"),
    Path("DATASET/TRAINING_SET/videos"), "TRAINING_SET")
print(bad.validate())
# ['onset 17.0s beyond duration 14.06s']
```

### `class AnnotationStore`
Tập hợp các `VideoAnnotation`, có thể truy vấn.

| Phương thức | Công dụng |
|---|---|
| `.from_split(dir)` | nạp toàn bộ một split (classmethod) |
| `.get("vid0012")` | tra theo id |
| `.labels` | list nhãn 0/1 — dùng cho chia fold phân tầng |
| `.filter(pred)` | lọc theo hàm điều kiện |
| `.subset(n)` | lấy n video, **giữ cân bằng dương/âm** |
| `.validated(drop=True)` | cảnh báo và loại bản ghi lỗi |
| `.describe()` | dict thống kê |

```python
from iwdd.annotations import AnnotationStore
s = AnnotationStore.from_split("DATASET/TRAINING_SET")
print(len(s))                                   # 400
print(len(s.filter(lambda a: a.time_of_day == "night")))   # 40
print(len(s.validated(drop=True)))              # 399  (loại vid0481)

night = s.subset(50)       # 25 dương + 25 âm, đan xen
```

`.subset()` đan xen dương/âm thay vì cắt 50 cái đầu — nếu cắt thẳng sẽ lệch
nhãn nặng vì file sắp theo id.

### Hàm phụ
- `_parse_resolution("1920x1080")` → `(1920, 1080)`, trả `(0,0)` nếu hỏng.
- `load_splits(root, ["TRAINING_SET", "VALIDATION_SET"])` → dict hai store.

---

## `iwdd/sampling.py` — trích frame ở 5 FPS

### `class SampledFrame` (frozen)
Một frame đã decode: `index` (vị trí trong chuỗi đã lấy mẫu), `source_index`
(vị trí trong video gốc), `timestamp` (giây), `image` (RGB uint8 HxWx3).

### `class VideoReader`
Context manager bọc `cv2.VideoCapture`.

```python
from iwdd.sampling import VideoReader
with VideoReader("DATASET/TRAINING_SET/videos/vid0001.mp4") as r:
    print(r.fps, r.frame_count)      # 30.0 173
```

**`.read_sequential()`** đọc tuần tự chứ không seek từng frame. Lý do: seek
bằng `CAP_PROP_POS_FRAMES` không đáng tin trên file VFR, mà corpus này có
nhiều file như vậy (20 mức fps khác nhau).

### `class FrameSampler`
Lấy mẫu đều ở `target_fps`.

**`.iter_sample(annotation)`** — generator, chỉ giữ 1 frame trong RAM.
Điểm then chốt là **stride thực**:

```python
stride = native_fps / target_fps      # vd 29.97 / 5 = 5.994
```

Stride là số thực nên phải **cộng dồn bằng float**; nếu làm tròn thành 5 thì
trên clip 109 giây sẽ lệch vài giây.

```python
from iwdd.sampling import FrameSampler
from iwdd.config import SamplingConfig
sampler = FrameSampler(SamplingConfig(target_fps=5.0))
frames = sampler.sample(a)                    # a = vid0001, 5.77s @30fps
print(len(frames), round(frames[-1].timestamp, 2))
# 29 5.6
```

**`.expected_num_frames(a)`** — ước lượng số frame **không cần mở file**, dùng
cho thanh tiến trình và lập chỉ mục window của `ClipDataset`.

**`._resize(image)`** — chỉ **thu nhỏ**, không phóng to. Corpus trải từ 320×240
đến 4096×2160; phóng to ảnh 320×240 chỉ tốn compute mà không thêm thông tin.

### `sampled_timestamps(annotation, target_fps)`
Dự đoán chuỗi timestamp mà sampler sẽ sinh ra, **không decode video**. Nhờ hàm
này mà `01_inspect_dataset.py` mô phỏng được phân bố nhãn trong 1 giây thay vì
phải giải mã 155 phút video.

```python
from iwdd.sampling import sampled_timestamps
ts = sampled_timestamps(a, 5.0)
print(ts.shape, ts[:4].round(2))    # (29,) [0.   0.2  0.4  0.6]
```

---

## `iwdd/labeling.py` — onset → nhãn từng frame

**File quan trọng nhất về mặt khoa học.** Dataset chỉ cho **một mốc onset**,
không có thời điểm kết thúc, nên cửa sổ dương là **do ta bịa ra**. Đây là lựa
chọn tự do ảnh hưởng lớn nhất tới kết quả.

### `class FrameLabels` (frozen)
`values` (mảng 0/1), `window` (cặp giây, `None` nếu âm tính), `clamped`
(True nếu cửa sổ bị cắt cho vừa video).

```python
.num_positive     # 10
.positive_ratio   # 0.345
```

### `class FrameLabeler`
**`.label(annotation, timestamps)`** — bung onset thành nhãn.

Quy tắc: frame dương khi `onset - pre ≤ t ≤ onset + post`.

```python
from iwdd.labeling import FrameLabeler
from iwdd.config import LabelingConfig
lab = FrameLabeler(LabelingConfig(post_seconds=2.0))
out = lab.label(a, ts)              # a.onset = 4.0
print(out.window, out.num_positive) # (4.0, 5.77) 9
print(out.clamped)                  # True  (video chỉ dài 5.77s)
```

Ba trường hợp biên được xử lý riêng:
1. **Cửa sổ rỗng** (như `vid0481`, onset nằm sau video) → đánh dấu frame cuối
   để video không thành "âm tính giả im lặng" khi huấn luyện.
2. **Cửa sổ lọt giữa hai frame** → gán cho frame gần onset nhất.
3. **Cửa sổ tràn quá video** → cắt về `duration`.

**`.positive_weight(label_batches)`** — tính `w+` thực tế từ dữ liệu.

```python
print(round(lab.positive_weight([out.values]), 2))   # 2.22
```

Trên toàn bộ 400 video, hàm này cho **21.3** chứ không phải 8.0 như bài báo —
xem `CLAUDE.md` §4.3.

---

# PHẦN C — Vision-Language

## `iwdd/captioning/prompts.py` — 5 câu hỏi

### `class Aspect(str, Enum)`
`SCENE`, `ACTIONS`, `OBJECTS`, `WASTE`, `DUMPING`.

### `class PromptSet` (frozen)
`prompts: dict[Aspect, str]` + `name`. Có `.aspects`, `.texts`, `len()`.

### Hằng số
- `MULTI_ASPECT_PROMPTS` — 5 prompt **nguyên văn** từ §3.2.2 bài báo.
- `SINGLE_PROMPT` — baseline `"a photo of"` (ablation Table 6).
- `get_prompt_set(single_prompt=False)` — chọn một trong hai.

```python
from iwdd.captioning import get_prompt_set, Aspect
ps = get_prompt_set()
print(len(ps))                          # 5
print(ps.prompts[Aspect.DUMPING][:45])  # Critical: Is anyone actively throwing
```

---

## `iwdd/captioning/base.py` — giao diện captioner

### `class CaptionSet`
5 caption của **một frame**.

| Phương thức | Công dụng |
|---|---|
| `.concat()` | nối 5 caption thành 1 chuỗi → chính là Eq. 6 của bài báo |
| `.ordered(aspects)` | trả list theo thứ tự aspect chỉ định |
| `.as_dict()` | serialize ra JSONL |

```python
from iwdd.captioning import CaptionSet, Aspect
cs = CaptionSet(frame_index=0, timestamp=0.0)
cs.texts[Aspect.SCENE] = "A street at night."
cs.texts[Aspect.DUMPING] = "No dumping activity observed."
print(cs.concat())
# A street at night. No dumping activity observed.
```

### `class BaseCaptioner(ABC)`
Hợp đồng chung. Lớp con bắt buộc cài `load()` và `generate()`.

| Phương thức | Vai trò |
|---|---|
| `.load()` | nạp trọng số (trừu tượng) |
| `.generate(images, prompt)` | caption 1 batch ảnh với **1** prompt (trừu tượng) |
| `.caption_frames(images, prompt_set)` | lặp 5 prompt × batch ảnh → list `CaptionSet` |
| `.embed_images(images)` | đặc trưng thị giác pooled, bỏ qua sinh text |
| `.unload()` | giải phóng VRAM |
| `._quantization_config()` | dựng `BitsAndBytesConfig` cho 8-bit/4-bit |

**Chi tiết hiệu năng quan trọng:** `.caption_frames()` lặp **prompt ở vòng
ngoài, frame ở vòng trong**. Nghĩa là mỗi aspect là một forward pass theo batch,
chứ không phải mỗi frame một lần gọi. Ngược lại sẽ chậm hơn nhiều lần.

### `class HFChatCaptioner(BaseCaptioner)`
Cài đặt dùng chung cho mọi model Hugging Face có chat template. Lớp con chỉ cần
khai báo `default_model_id` và `model_class_name`.

| Phương thức nội bộ | Việc |
|---|---|
| `._build_chat_text(prompt)` | dựng 1 lượt user có ảnh + câu hỏi |
| `._find_vision_tower()` | dò `vision_model` / `visual` / ... qua 6 layout |
| `._move(value)` | chuyển tensor sang device đúng dtype |
| `._clean(text)` | gộp khoảng trắng, bỏ tiền tố `"Assistant:"` |

Khi `load()`, nó đặt `tokenizer.padding_side = "left"` — bắt buộc, vì pad bên
phải sẽ làm lệch phần sinh ra giữa các mẫu trong batch.

### `class DummyCaptioner`
Giả lập không cần trọng số, dùng smoke-test khi không có GPU. `generate()` trả
caption từ độ sáng trung bình; `embed_images()` trả histogram màu 48 chiều.

```python
import numpy as np
from iwdd.captioning import DummyCaptioner
d = DummyCaptioner(); d.load()
img = np.full((64, 64, 3), 128, np.uint8)
print(d.generate([img], "Describe this scene.")[0])
# Describe this scene | mean brightness 128.0
print(d.embed_images([img]).shape)    # (1, 48)
```

---

## `iwdd/captioning/backends.py` — các VLM cụ thể

| Lớp | `default_model_id` | Ghi chú |
|---|---|---|
| `SmolVLMCaptioner` | `HuggingFaceTB/SmolVLM-500M-Instruct` | mặc định, hợp edge |
| `SmolVLM2Captioner` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | bản kế nhiệm |
| `Qwen3VLCaptioner` | `Qwen/Qwen3-VL-8B-Instruct` | mạnh nhất, ~17GB bf16 |
| `CosmosNanoCaptioner` | `nvidia/Cosmos-Reason1-7B` | **chưa xác minh id** |
| `BlipCaptioner` | `Salesforce/blip-image-captioning-large` | VLM tốt nhất trong bài báo |

`CosmosNanoCaptioner.load()` tự bật 8-bit nếu chưa chọn độ chính xác nào.

`BlipCaptioner` **không kế thừa** `HFChatCaptioner` vì BLIP không có chat
template — nó nhận tiền tố text rồi viết tiếp, nên 5 prompt được dùng làm
prefix điều kiện.

### Factory
```python
from iwdd.captioning import build_captioner, available_backends
from iwdd.config import CaptionConfig
print(available_backends())
# ['blip', 'cosmos_nano', 'dummy', 'qwen3vl', 'smolvlm', 'smolvlm2']
cap = build_captioner(CaptionConfig(backend="smolvlm"))
print(cap.model_id)      # HuggingFaceTB/SmolVLM-500M-Instruct
```

---

## `iwdd/encoders.py` — caption → vector

### `class EncoderMode(str, Enum)`
| Mode | Làm gì | Số chiều |
|---|---|---|
| `CONCAT_TEXT` | nối 5 caption, encode 1 lần — **đúng bài báo** | 384 |
| `PER_ASPECT` | encode riêng từng aspect rồi nối | 1920 |
| `VISION` | bỏ hẳn text, pool vision tower | tùy model |

### `class BaseFrameEncoder(ABC)`
Giao diện: `.output_dim`, `.load()`, `.encode(caption_sets, images, prompt_set)`.

### `class SentenceBertEncoder`
Phục vụ hai mode text. Dùng `all-MiniLM-L6-v2` (384-d, khớp Eq. 6).

```python
# (cần sentence-transformers — chưa cài trên máy này)
from iwdd.encoders import SentenceBertEncoder
from iwdd.config import EncoderConfig
enc = SentenceBertEncoder(EncoderConfig(mode="concat_text"))
emb = enc.encode(caption_sets, [], ps)
print(emb.shape)      # (29, 384)    — 29 frame
```

Với `mode="per_aspect"`, nó "làm phẳng" 29×5 = 145 câu, encode một lượt rồi
reshape về `(29, 1920)` — nhanh hơn encode từng frame.

### `class VisionFeatureEncoder`
Bọc một captioner, gọi `.embed_images()`. Đây là đường cắt bỏ 90% chi phí.

### Hàm
- `build_encoder(config, captioner)` — factory; mode `vision` **bắt buộc** có captioner.
- `aspect_slice(aspect, prompt_set, base_dim)` — lấy cột của một aspect trong
  vector `per_aspect`, để ablation mà **không phải caption lại**.

```python
from iwdd.encoders import aspect_slice
from iwdd.captioning import Aspect, MULTI_ASPECT_PROMPTS
print(aspect_slice(Aspect.WASTE, MULTI_ASPECT_PROMPTS, 384))
# slice(1152, 1536, None)    — aspect thứ 4
```

---

## `iwdd/features.py` — điều phối + cache

### `class FrameFeatures`
Kết quả của một video: `embeddings (N,D)`, `timestamps (N,)`,
`frame_labels (N,)`, `video_label`, `onset`, `duration`, `modality`.

```python
.num_frames      # 29
.feature_dim     # 384
.save(path)      # ghi .npz nén
FrameFeatures.load(path)   # đọc lại
```

### `class FeatureCache`
Thư mục chứa một `.npz` mỗi video + `manifest.json`.

```python
from iwdd.features import FeatureCache
c = FeatureCache("artifacts/features/dummy-default-multi-full-fps5-vision")
print(c.has("vid0001"))            # True
f = c.load("vid0001")
print(f.num_frames, f.feature_dim) # 29 48
items = c.load_many(["vid0001", "vid0002"])   # thiếu thì cảnh báo, không crash
```

`manifest.json` lưu lại toàn bộ tham số đã sinh ra cache — để sau này biết
cache đó từ đâu ra.

### `class FeatureExtractionPipeline`
Nối giai đoạn 1–3. Dựng sẵn sampler, labeler, captioner, encoder, cache.

**`.run(store, overwrite=False)`** — bỏ qua video đã cache, bắt exception từng
video (một file hỏng không được làm sập một job 15 tiếng), ghi manifest cuối.

**`.process_video(a)`** — xử lý một video: sample → caption → encode → label.

```python
from iwdd.features import FeatureExtractionPipeline
from iwdd.config import ExperimentConfig
cfg = ExperimentConfig()
cfg.caption.backend = "dummy"; cfg.encoder.mode = "vision"
pipe = FeatureExtractionPipeline(cfg)
ids = pipe.run(store.subset(6))
print(ids)              # ['vid0001', 'vid0002', ...]
print(pipe.probe.fps)   # 31.98
```

### `relabel_cache(cache, store, labeler)`
**Hàm tiết kiệm nhất cả repo.** Ghi lại nhãn frame mà **không caption lại** —
nhờ vậy sweep `post_seconds` tốn vài giây thay vì 15 giờ.

```python
from iwdd.features import relabel_cache
relabel_cache(c, store, FrameLabeler(LabelingConfig(post_seconds=4.0)))
# INFO | Relabelled 400 cached videos
```

---

# PHẦN D — Dataset PyTorch và mô hình

## `iwdd/datasets.py`

### `class SequenceBatch`
Batch đã pad: `features (B,T,D)`, `labels (B,T)`, `mask (B,T)`,
`timestamps (B,T)`, `video_labels (B,)`, `onsets (B,)`, `video_ids`.
`.to(device)` chuyển toàn bộ tensor.

**Vì sao cần `mask`:** clip dài từ 2.9s đến 109.5s → 14 đến 548 frame. Không có
mask thì phần pad sẽ chi phối loss.

### `class SequenceDataset`
Mỗi item là một `FrameFeatures`. Constructor **raise nếu số chiều không đồng
nhất** — chặn lỗi trộn cache của hai VLM khác nhau.

```python
from iwdd.datasets import SequenceDataset
ds = SequenceDataset.from_cache(c, store)
print(len(ds), ds.feature_dim)             # 40 48
print(round(ds.positive_frame_weight(), 2)) # 19.72  ← w+ thật
sub = ds.subset([0, 1, 2])                  # tạo view theo chỉ số (dùng cho CV)
```

### `class ClipDataset`
Cho 3D CNN: mỗi item là một cửa sổ `clip_len` frame ảnh thô.
Nhãn cửa sổ = 1 nếu **bất kỳ** frame nào trong đó nằm trong cửa sổ sự kiện.

`.windows` là list phẳng `(vị_trí_video, frame_bắt_đầu)`, lập chỉ mục bằng
`expected_num_frames()` nên **không cần decode lúc khởi tạo**.

`._frames_for()` có **LRU cache có giới hạn** — xem `CLAUDE.md` §6: bản đầu
decode lại cả video cho **mỗi** cửa sổ, 707 lần thay vì 100 lần.

### Các hàm
| Hàm | Công dụng |
|---|---|
| `collate_sequences(items)` | pad chuỗi dài ngắn khác nhau về `SequenceBatch` |
| `build_sequence_loader(ds, bs, shuffle)` | DataLoader cho mô hình chuỗi |
| `build_clip_loader(ds, bs, shuffle)` | DataLoader cho 3D CNN |
| `stratified_folds(labels, k, seed)` | chia fold **cân bằng nhãn** |
| `discover_cache_dim(root)` | đọc số chiều từ file `.npz` đầu tiên |

```python
from iwdd.datasets import stratified_folds
folds = stratified_folds([0,1]*20, num_folds=5, seed=42)
tr, va = folds[0]
print(len(tr), len(va))     # 32 8
```

---

## `iwdd/models/base.py`

### `class BaseTemporalModel(nn.Module, ABC)`
Gốc chung. `input_kind` là `"sequence"` hoặc `"clip"` — vòng huấn luyện **điều
phối dựa vào thuộc tính này**.

```python
.num_parameters   # đếm tham số trainable
.describe()       # {'class': ..., 'input_kind': ..., 'num_parameters': ...}
```

### `class BaseSequenceModel`
Lớp con chỉ cần cài `encode(features, mask) -> (B,T,H)`. Phần `forward()` đã lo:

```python
logits = head(dropout(norm(hidden)))
return logits.masked_fill(~mask, -30.0)   # ép vị trí pad xuống rất thấp
```

**`._set_head(output_dim)`** tạo `LayerNorm` + `Linear`. `LayerNorm` không phải
trang trí — thiếu nó thì hụt **đúng 1024 tham số** so với bài báo (xem
`CLAUDE.md` §5).

### `class BaseClipModel`
`forward(clips) -> (B,)` — một logit mỗi cửa sổ.

### Hàm
- `masked_mean(values, mask)` — trung bình theo thời gian, bỏ qua pad.
- `lengths_from_mask(mask)` — mask → độ dài thật, tối thiểu 1.

```python
import torch
from iwdd.models.base import lengths_from_mask
m = torch.tensor([[1,1,1,0],[1,1,0,0]], dtype=torch.bool)
print(lengths_from_mask(m))    # tensor([3, 2])
```

---

## `iwdd/models/recurrent.py` — 5 kiến trúc của bài báo

### `class _RecurrentCore` (nội bộ)
Bọc `nn.LSTM`/`nn.GRU` kèm pack/pad, để frame pad **không** ảnh hưởng hidden state.

```python
packed = pack_padded_sequence(features, lengths, enforce_sorted=False)
output, _ = self.rnn(packed)
hidden, _ = pad_packed_sequence(output, total_length=T)
```

`hidden_dim` là **độ rộng mỗi chiều** (chuẩn PyTorch), nên đầu ra là `2 × hidden_dim`.

### `class _Conv1dFrontEnd` (nội bộ)
`Conv1d → BatchNorm → ReLU → Dropout`, padding giữ nguyên độ dài. Bắt biến đổi
cục bộ giữa các caption kề nhau — quan trọng vì dumping động có thể xong trong
dưới một giây.

### 5 lớp công khai

| Lớp | Cấu trúc | Tham số @384-d |
|---|---|---|
| `LSTMClassifier` | BiLSTM | 2,893,313 ✓ khớp bài báo |
| `GRUClassifier` | BiGRU | 2,170,369 ✓ khớp bài báo |
| `CNNLSTMClassifier` | Conv1D → BiLSTM | 2,926,849 |
| `CNNGRUClassifier` | Conv1D → BiGRU | 2,269,441 ← **tốt nhất bài báo** |
| `BiLSTMAttentionClassifier` | BiLSTM + cổng attention | 2,893,826 |

```python
import torch
from iwdd.config import ModelConfig
from iwdd.models import build_model
m = build_model(ModelConfig(name="cnn_gru"), input_dim=384)
x = torch.randn(2, 50, 384)
mask = torch.zeros(2, 50, dtype=torch.bool); mask[0,:50]=True; mask[1,:14]=True
out = m(x, mask)
print(out.shape, bool((out[1,14:] < -20).all()))
# torch.Size([2, 50]) True        ← vị trí pad đã bị ép xuống
```

Trong `BiLSTMAttentionClassifier`, trọng số attention được **nhân lại với T** để
cổng chỉ tái phân bổ trọng số giữa các frame, không làm co activation về 0 trên
chuỗi dài.

---

## `iwdd/models/transformer.py`

### `class SinusoidalPositionalEncoding`
Bảng vị trí cố định, `max_len=2048` (corpus cần ~550 @5FPS). **Raise nếu chuỗi
dài hơn bảng** thay vì cắt âm thầm.

### `class TransformerClassifier`
`Linear → PosEnc → TransformerEncoder → LayerNorm → Linear`.
Dùng `src_key_padding_mask=~mask` — PyTorch quy ước **True = bỏ qua**, ngược với
mask của ta, nên phải phủ định.

Bài báo xếp kiến trúc này kém nhất (F1 0.15–0.31) vì ba lý do: thiếu dữ liệu,
attention toàn cục không hợp sự kiện cục bộ, và onset xuất hiện ở vị trí bất kỳ.

---

## `iwdd/models/cnn3d.py` — đối chứng pixel

Không có trong bài báo. Đây là phép đối chứng mà bài báo thiếu: mọi kiến trúc
khác chỉ nhìn video **qua caption**, nên không cái nào trả lời được câu
"vòng captioning có đáng 1050ms/frame không".

| Lớp | Mô tả |
|---|---|
| `_Conv3DBlock` | `Conv3d → BatchNorm3d → ReLU → MaxPool3d` |
| `Conv3DClassifier` | 4 block, huấn luyện từ đầu — 1,165,025 tham số |
| `PretrainedR3DClassifier` | R3D-18 pretrain Kinetics-400, thay head |

`PretrainedR3DClassifier` có `freeze_backbone` — với 400 video thì đóng băng
backbone là điểm khởi đầu hợp lý.

---

## `iwdd/models/__init__.py` — registry

```python
from iwdd.models import (MODEL_REGISTRY, PAPER_ARCHITECTURES,
                         build_model, available_architectures, input_kind_of)
print(available_architectures())
# ['bilstm_attention','cnn_gru','cnn_lstm','conv3d','gru','lstm','r3d18','transformer']
print(PAPER_ARCHITECTURES)
# ('lstm','cnn_lstm','gru','cnn_gru','bilstm_attention','transformer')
print(input_kind_of("conv3d"))     # clip
```

`build_model(config, input_dim)` tự phân nhánh: mô hình clip bỏ qua `input_dim`,
mô hình chuỗi **raise nếu `input_dim <= 0`**.

---

# PHẦN E — Quyết định và chấm điểm

## `iwdd/postprocess.py` — từ điểm số thành một cảnh báo

### `class DetectionResult` (frozen)
`detected`, `onset`, `peak_score`, `peak_time`, `run_length`, `smoothed`.

### `class EventDetector`
**`.smooth(scores)`** — trung bình trượt cửa sổ 9, **phản xạ ở biên** thay vì
pad 0. Lý do: có một video train onset ở t=0, pad 0 sẽ dập tắt nó.

**`.detect(scores, timestamps)`** — ngưỡng kép:
1. đỉnh đã làm mượt phải `> 0.7`, **và**
2. phải có `≥ 5` frame liên tiếp `> 0.6` (ở 5 FPS = một giây bằng chứng liên tục).

```python
import numpy as np
from iwdd.postprocess import EventDetector
det = EventDetector()
t = np.arange(0, 20, 0.2, dtype=np.float32)
s = np.zeros_like(t); s[25:40] = 0.95        # chuỗi mạnh bắt đầu t=5.0s
r = det.detect(s, t)
print(r.detected, round(r.onset, 2), round(r.peak_score, 3))
# True 5.2 0.95
```

Onset ra 5.2 chứ không phải 5.0 vì làm mượt đẩy điểm bắt đầu chuỗi lùi một chút —
đúng như thiết kế, và độ trễ 0.2s được tính vào metric.

**`._longest_run(flags)`** — tìm chuỗi `True` dài nhất, trả `(start, length)`.

### `tune_thresholds(...)`
Grid-search `peak_threshold` × `run_threshold` theo F1 contest. Bài báo cố định
0.7/0.6 mà không nói chọn thế nào — đó là giá trị tune cho BLIP, nên VLM khác
gần như chắc chắn cần giá trị khác. **Chỉ tune trên fold huấn luyện.**

---

## `iwdd/metrics.py` — luật chấm của IWDD Contest 2026

### `class ConfusionCounts`
`true_positive`, `false_positive`, `false_negative`, `true_negative` +
các property `precision`, `recall`, `f1`, `accuracy`, `specificity`.

### `class EvaluationReport`
`counts`, `delays`, `num_videos`, `per_video` (dùng để phân tích lỗi).

```python
.mean_delay                       # D  = trung bình |p - g| trên các TP
.normalized_delay(T_max)          # max(0, 1 - D/T_max)
.summary(config)                  # dict phẳng mọi chỉ số
```

### `class IWDDEvaluator`
**`.is_valid_detection(p, g)`** — hợp lệ khi `g - 3.0 ≤ p ≤ g + 10.0`.

```python
from iwdd.metrics import IWDDEvaluator
from iwdd.config import MetricConfig
ev = IWDDEvaluator(MetricConfig())
print(ev.is_valid_detection(5.0, 5.0))   # True
print(ev.is_valid_detection(5.0, 1.0))   # True   (trễ 4s, trong +10)
print(ev.is_valid_detection(5.0, 9.0))   # False  (sớm 4s, vượt -3)
```

**`._score_one()`** — điểm tinh tế nhất của cả repo:

```python
# Video DƯƠNG mà báo động SAI THỜI ĐIỂM  →  tính CẢ FP LẪN FN
# Trả về chuỗi outcome "FP+FN".
```

Đó là cách đọc đúng nghĩa đen của §3.4, và cũng là động cơ đúng: một cảnh báo
sai thời điểm vừa là báo động giả, vừa là bỏ sót — không phải nửa điểm.

**`.processing_score(fps)` / `.memory_score(gb)`** — điểm phạt throughput và bộ
nhớ; `0` nghĩa là đạt chỉ tiêu triển khai.

### `aggregate_folds(reports, config)`
Trung bình + **độ lệch chuẩn** qua các fold. Luôn trả cả hai, vì với 10 video
mỗi fold thì độ lệch quan trọng ngang giá trị trung bình (bài báo: 0.7429 ± 0.191).

---

# PHẦN F — Huấn luyện

## `iwdd/training.py`

### `class VideoPrediction`
Đường cong điểm của một video: `video_id`, `scores`, `timestamps`, `onset`,
`video_label`.

### `class FoldResult`
Kết quả một fold: `report`, `best_epoch`, `history`, `num_parameters`,
`checkpoint_path`.

### `class MaskedWeightedBCE(nn.Module)`
BCE có trọng số, **bỏ qua frame pad**.

```python
total = (per_frame * weights * valid).sum()
return total / (weights * valid).sum().clamp_min(1.0)
```

Chia cho **tổng trọng số hợp lệ của cả batch**, không phải trung bình theo từng
chuỗi — nếu không, video 109s sẽ nặng gấp 38 lần video 2.9s.

### `class Trainer`
| Phương thức | Việc |
|---|---|
| `.fit(train, val)` | 20 epoch, Adam 1e-3, ReduceLROnPlateau, clip 1.0; **giữ trọng số tốt nhất** |
| `.predict(loader)` | sinh đường cong điểm từng video, đo throughput |
| `.evaluate(loader)` | predict → detect → chấm điểm, trả `(report, results)` |
| `._run_epoch(loader, optim)` | một epoch; `optim=None` nghĩa là chế độ eval |

```python
from iwdd.training import Trainer
tr = Trainer(cfg, model)
history, best_epoch, best_state = tr.fit(train_loader, val_loader)
report, results = tr.evaluate(val_loader)
print(report.summary(cfg.metric)["f1"])
```

### `class CrossValidator`
`.run(save_checkpoints=True)` — k-fold phân tầng, **reseed mỗi fold**
(`seed + fold`), lưu checkpoint tốt nhất, trả `(list[FoldResult], aggregated)`.

### `class BenchmarkRunner`
`.run(architectures)` — quét nhiều kiến trúc trên **cùng một cache đặc trưng**.
Vì VLM đã đóng băng và cache sẵn, cả 6 kiến trúc chỉ tốn vài phút.
`._write_table()` ghi `benchmark.json` + `benchmark.md`.

```python
from iwdd.training import BenchmarkRunner
from iwdd.models import PAPER_ARCHITECTURES
table = BenchmarkRunner(cfg, ds).run(PAPER_ARCHITECTURES)
```

---

## `iwdd/clip_training.py` — nhánh 3D CNN

Tách riêng vì hợp đồng dữ liệu khác: clip model nhận pixel thô và trả **một
logit mỗi cửa sổ**, trong khi sequence model trả **một logit mỗi frame**.

### `class ClipTrainer`
**`.predict(loader, store)`** — chỗ thú vị nhất: rải điểm của từng clip trở lại
trục thời gian.

```python
centre = (start_time + end_time) / 2      # mỗi clip đóng góp tại tâm của nó
buckets[video_id].append((centre, score)) # gom theo video, sắp theo thời gian
```

Nhờ vậy mà 3D CNN đi qua **đúng cùng một `EventDetector` và `IWDDEvaluator`**
với các mô hình khác — so sánh mới công bằng.

### `run_clip_experiment(config, train_store, test_store)`
Hàm tiện ích: dựng `ClipDataset`, loader, model, huấn luyện, đánh giá, lưu
checkpoint. Raise `TypeError` nếu tên kiến trúc không phải clip model.

---

# PHẦN G — Script dòng lệnh

## `scripts/01_inspect_dataset.py`
**Chạy đầu tiên, luôn luôn.** Kiểm schema thật, in thống kê, liệt kê bản ghi lỗi.

- `parse_args()` — `--dataset-root`, `--splits`, `--target-fps`, `--post-seconds`, `--out`
- `label_statistics(store, fps, labeler)` — **mô phỏng** phân bố nhãn bằng
  `sampled_timestamps()`, không decode video nào. Đây là hàm phát hiện ra
  `w+ = 8` không khớp cửa sổ 2 giây.

```bash
.\.venv\Scripts\python.exe scripts/01_inspect_dataset.py --post-seconds 2.0
#   positive_ratio             0.0448
#   implied_positive_weight    21.34
```

## `scripts/02_extract_features.py`
Giai đoạn đắt nhất, chạy **một lần mỗi VLM**.

- `build_config(args)` — gom CLI thành `ExperimentConfig`
- `main()` — in signature, nạp store, `.validated()`, chạy pipeline, in throughput

```bash
# kiểm tra đường ống, không cần GPU
... scripts/02_extract_features.py --backend dummy --encoder-mode vision --limit 6
# chạy thật
... scripts/02_extract_features.py --backend smolvlm
```

## `scripts/03_train.py`
Entry point chính, **tự phân nhánh theo `input_kind`**.

| Hàm | Việc |
|---|---|
| `load_sequence_dataset(cfg, split, sig, subset)` | nạp cache, raise nếu thiếu |
| `run_sequence(cfg, args)` | holdout hoặc `--cv` |
| `run_clip(cfg)` | nhánh 3D CNN |
| `main()` | lưu `config.json` + `results.json` |

Cờ `--auto-positive-weight` thay hằng số 8.0 của bài báo bằng giá trị suy từ dữ liệu.

```bash
... scripts/03_train.py --arch cnn_gru --cv --folds 5 --subset 50 --signature <sig>
... scripts/03_train.py --arch conv3d --clip-len 16
```

## `scripts/04_benchmark.py`
Quét toàn bộ kiến trúc trên một cache. Nó **import lại `load_sequence_dataset`
từ `03_train.py`** qua `importlib` (vì tên file bắt đầu bằng số nên không
import thường được).

```bash
... scripts/04_benchmark.py --signature <sig> --folds 5
```

---

# Phụ lục — ba quy ước xuyên suốt

**1. Mask là True cho frame thật.** PyTorch Transformer quy ước ngược lại, nên
chỗ duy nhất phủ định là `src_key_padding_mask=~mask`.

**2. Mọi thứ đắt đều được cache, khóa bằng signature.** VLM đóng băng → caption
chỉ sinh một lần. Đổi VLM → signature đổi → cache không đụng nhau.

**3. Số âm tính chuẩn là `onset = -1`, không phải `None`,** khi đã vào tensor.
`VideoAnnotation.onset` dùng `None`; từ `FrameFeatures` trở đi dùng `-1.0` để
lưu được vào `.npz` và tensor.
