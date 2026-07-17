# I-JEPA: Image-based Joint-Embedding Predictive Architecture

A from-scratch PyTorch implementation of [I-JEPA](https://arxiv.org/abs/2301.08243) (Assran et al., CVPR 2023). I-JEPA is a self-supervised learning method for vision: it predicts latent representations of masked target image patches from visible context, without hand-crafted data augmentations. The core idea is learning semantic representations by predicting *in representation space* rather than in pixel space.

This implementation follows the reference architecture closely — pre-norm ViT encoder, EMA target encoder, narrower ViT predictor, multi-block masking, per-block prediction, and smooth-L1 loss — and is built to train on both small (STL-10) and large (ImageNet-1k) datasets from a single GPU.

---

## Architecture

I-JEPA has three components around a shared ViT backbone:

```
     Input Image
         │
    ┌────┴────┐
    │  Conv   │  patch embedding (kernel = stride = patch_size)
    │ + Pos   │  learnable positional embeddings
    └────┬────┘
         │
    ┌────┼────────────┐
    ▼    ▼            ▼
Context  Target       Raw
Encoder  Encoder      Image
 (θ)     (θ̄ = EMA(θ))
   │       │
   │       ▼ (stop-grad, no_grad)
   │    layer_norm
   │    slice target blocks
   │       │
   ▼       ▼
Predictor   Target repr.
 (φ)         (ground truth)
   │         │
   └────┬────┘
        ▼
   smooth_l1_loss
```

**Context encoder** `f_θ` — a ViT that sees only the context (visible) patches. Target patches are dropped via `torch.gather` *before* the transformer, so attention never touches them.

**Target encoder** `f_θ̄` — a `deepcopy` of the context encoder, updated by **EMA** of the context encoder's weights (`θ̄ ← m·θ̄ + (1−m)·θ`, momentum ramped 0.996→1.0). Never touched by backprop. Sees the **full, uncorrupted image** so every target-patch representation is contextualised by the entire scene.

**Predictor** `g_φ` — a narrower, shallower ViT. Takes the context encoder's output, inserts a single learnable mask token at each target-patch position with that position's embedding, then runs **per-block** prediction: the context is replicated once per block and the block axis is folded into the batch axis, so each block's mask tokens attend to the context and to each other but never to another block's mask tokens.

**Loss** — `F.smooth_l1_loss(predictions, targets)`, matching the [reference implementation](https://github.com/facebookresearch/ijepa). The paper describes "average L2 distance" but the released code uses smooth L1 — a known paper-vs-code discrepancy.

**Multi-block masking** — 4 target rectangles (scale 0.15–0.2, aspect 0.75–1.5, independently placed; may overlap) and 1 context rectangle (scale 0.85–1.0). Context = context_block − union(targets), so they are disjoint. Block sizes shared across the batch; positions vary per image.

---

## Project structure

```
jepa-exp/
├── configs/
│   ├── stl10/                          STL-10 configs (small and medium models)
│   │   ├── stl10_ijepa.yaml
│   │   ├── stl10_ijepa_m.yaml
│   │   ├── stl10_ijepa_classifier.yaml
│   │   └── stl10_ijepa_m_classifier.yaml
│   └── imagenet/                       ImageNet-1k configs (ViT-B/16)
│       ├── imagenet_ijepa.yaml
│       └── imagenet_ijepa_classifier.yaml
│
├── src/
│   ├── models/
│   │   ├── modules/
│   │   │   ├── components.py           MHA, SwiGLU FFN, RMSNorm, Block
│   │   │   ├── encoder.py              ViTEncoder
│   │   │   └── predictor.py            ViTPredictor
│   │   ├── ijepa.py                    I-JEPA wrapper (context + EMA target + predictor)
│   │   └── classifier.py              Frozen encoder + linear head
│   │
│   ├── datasets/
│   │   ├── stl10_unlabelled_dataset.py     STL-10 unlabelled (100k), uint8 memmap
│   │   ├── stl10_labelled_dataset.py       STL-10 labelled (5k), uint8 memmap
│   │   ├── imagenet_stream_dataset.py      ImageNet-1k, streaming from HF
│   │   └── imagenet_cached_dataset.py      ImageNet-1k, local parquet cache
│   │
│   ├── training/
│   │   ├── unlabelled_trainer.py           I-JEPA pretraining loop
│   │   └── labelled_trainer.py             Linear-probe training loop
│   │
│   └── utils/
│       ├── masking.py                  Multi-block masking (per the reference MaskCollator)
│       ├── metrics.py                  ThroughputMonitor, param counts
│       └── performance.py              accuracy, macro-F1
│
└── scripts/
    ├── stl10_download.py                        Download STL-10
    ├── unlabelled_training.py                   Pretrain on STL-10 (memmap)
    ├── labelled_training.py                     Linear probe on STL-10 (memmap)
    ├── imagenet_download.py                     Verify / cache / materialize ImageNet
    ├── imagenet_streaming_unlabelled_training.py   Pretrain on ImageNet
    ├── imagenet_streaming_labelled_training.py     Linear probe on ImageNet
    └── predict.py                               Inference from a checkpoint
```

---

## Quick start

```bash
# 1. Set up
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision datasets huggingface_hub pillow pyyaml wandb python-dotenv

# 2. Download STL-10 for fast iteration
python -m scripts.stl10_download

# 3. Pretrain the small model on STL-10
python -m scripts.unlabelled_training --config configs/stl10/stl10_ijepa.yaml

# 4. Linear probe on the pretrained encoder
python -m scripts.labelled_training --config configs/stl10/stl10_ijepa_classifier.yaml
```

For ImageNet-1k (ViT-B/16, 256×256):

```bash
# Cache the compressed parquet (~62 GB for all splits, one time)
python -m scripts.imagenet_download --cache

# Pretrain
python -m scripts.imagenet_streaming_unlabelled_training --config configs/imagenet/imagenet_ijepa.yaml

# Linear probe
python -m scripts.imagenet_streaming_labelled_training --config configs/imagenet/imagenet_ijepa_classifier.yaml
```

Create a `.env` file with `HF_TOKEN=your_token` to authenticate HuggingFace Hub requests (`load_dotenv` at the top of every script loads it automatically).

---

## Datasets

All datasets return `(C, H, W)` float tensors in `[0, 1]`.

### STL-10

| Split | Images | Resolution | Format |
|---|---|---|---|
| Unlabelled | 100,000 | 96×96 | Local uint8 memmap |
| Train (labelled) | 5,000 | 96×96 | Local uint8 memmap |

Used for development and fast iteration. The small model saturates at ~48% probe accuracy — the dataset is too small for I‑JEPA's data-hungry training regime, but it catches implementation bugs quickly.

### ImageNet-1k

| Split | Images | Resolution | Format |
|---|---|---|---|
| Train | 1,281,167 | 256×256 | Parquet on HF Hub, cached locally |
| Val | 50,000 | 256×256 | Same |
| Test | 100,000 | 256×256 | Same (labels unused) |

Two options, controlled by `use_streaming` in the config:

- **Cached (`use_streaming: false`, default).** `ImageNetCachedDataset` is a map-style `Dataset` over the parquet downloaded to `~/.cache/huggingface`. Full random access: `shuffle=True`, any `num_workers`, `persistent_workers`. All network failures and shard caps simply don't exist. Run `imagenet_download.py --cache` once.
- **Streaming (`use_streaming: true`).** `ImageNetStreamDataset` is an `IterableDataset` pulling shards on demand. Zero local storage but caps at 5 workers (`.shuffle()` collapses the 52 shards to 5) and can hit HTTP-client lifetime bugs on worker reuse.

Images go through a `resize_center_crop` transform: short side resized to `image_size` (preserving aspect ratio), then square center-crop, forced to RGB. The source mirror keeps the short side at 256 but the long side varies — e.g. `341×256` — so a naive `resize((256,256))` squashes aspect ratios.

---

## Training

### I-JEPA pretraining

The pretraining loop (`unlabelled_trainer.py`):

1. **MultiBlockMasking** samples a disjoint context/target split on the patch grid every step.
2. **Context encoder** processes the context patches only → `context_repr`.
3. **Target encoder** processes the full image (`no_grad`, EMA weights) → `F.layer_norm` normalises each token → slice out the target-block patches → **targets**.
4. **Predictor** inserts mask tokens at target positions with positional embeddings, runs a per-block forward, projects back to `encoder_dim` → **predictions**.
5. `smooth_l1_loss(predictions, targets)`, averaged over blocks and patches.

**EMA momentum** follows a cosine schedule from `ema_start` → `ema_end`. **Learning rate** follows linear warmup + cosine decay to 10% of max. Gradients are clipped per optimisation step (max norm 1.0). Loss is divided by `gradient_accumulation_steps` so the effective gradient is the mean.

#### Model scales

| | Small | ViT-Base |
|---|---|---|
| Encoder dim | 256 | 768 |
| Encoder FF | 512 | 3072 |
| Encoder heads / layers | 4 / 6 | 12 / 12 |
| Predictor dim | 128 | 384 |
| Predictor FF / heads / layers | 256 / 4 / 3 | 1536 / 12 / 6 |
| Encoder params | ~4M | ~114M |
| Typical dataset | STL-10 | ImageNet-1k |

### Linear probe

The classifier (`classifier.py`) freezes the encoder and trains a **single linear layer** on mean-pooled token representations — exactly as the paper describes: average-pool the target-encoder output. Gradients flow only through the classifier head.

The probe trainer (`labelled_trainer.py`) reports loss, accuracy, and macro-F1 on the full validation split at each log interval. Validation can be capped at `val_max_batches` batches to keep eval fast on large val sets (ImageNet val is 50k images).

### Checkpointing and resume

Checkpoints save the full model (`context_encoder` + `target_encoder` + `predictor`), optimizer state, mask-RNG generator state, step count, and config — everything needed to resume faithfully or warm-start a probe.

- `from_checkpoint` alone: warm-starts the encoder weights, fresh optimizer, step 0.
- `from_checkpoint` + `is_resume: true`: restores optimizer, step, and mask RNG for a faithful continuation of the same run.

---

## Configuration

Key parameters in the YAML configs:

### Pretraining

| Parameter | Description |
|---|---|
| `image_size` | Input resolution (must be divisible by `patch_size`) |
| `patch_size` | Patch-embedding conv kernel/stride; drives `grid_size` and `max_seq_len` |
| `batch_size` | Per-GPU batch |
| `gradient_accumulation_steps` | Effective batch = `batch_size × this` |
| `learning_rate` | Peak LR after warmup |
| `warmup_steps` | Linear LR warmup duration |
| `max_steps` | Total training steps |
| `weight_decay` | AdamW weight decay |
| `ema_start / ema_end` | Target-encoder momentum schedule (cosine) |
| `num_target_blocks` | Number of target rectangles (4 in the paper) |
| `log_every / save_every` | Step intervals for logging and checkpointing |

### Linear probe

| Parameter | Description |
|---|---|
| `from_checkpoint` | Path to an I-JEPA pretrain checkpoint for encoder warm-start |
| `encoder_source` | `"target_encoder"` (paper default) or `"context_encoder"` |
| `is_resume` | Restore optimizer + step for continued training |
| `val_max_batches` | Cap validation at N batches (default: full pass) |
| `use_streaming` | `false` (cached, default) or `true` (stream from HF) |

---

## Implementation notes

**Encoder final norm.** Pre-norm transformer blocks return the raw residual stream — nothing normalises it, and its variance grows with depth. A final `RMSNorm` is both the architectural standard (every ViT/GPT/LLaMA has one) and a correctness requirement here: without it, the context encoder's output has σ ≈ 13 while the target representations are `F.layer_norm`-standardised to σ ≈ 1, a 12× scale mismatch the predictor's projections must absorb through their weights.

> This is separate from the per-token `F.layer_norm` applied to the target encoder's output — that one is not affine (no learnable parameters) and exists purely to standardise the regression target, discouraging the collapse solution of simply shrinking the encoder's output. Two norms, two jobs.

**No augmentations.** I-JEPA deliberately avoids hand-crafted invariances — the only source of input variation is the random block masking. This removes the augmentation prior, which is the method's conceptual strength, but also makes it **data-hungry**: 100k STL images cap a 4M encoder at ~48% probe accuracy, and a 114M ViT-B gets the same score on the same data. Model scaling only pays off once data is abundant.

**Pretraining loss is not a progress signal.** Because the target encoder (EMA) moves every step, the loss is a moving-goalpost regression: it drops quickly at first, then enters a flat, volatile plateau that drifts only slowly. You cannot tell from the loss curve alone whether training has stalled or is steadily improving. Downstream **probe accuracy over checkpoints** is the correct diagnostic. The best checkpoint is often not the last one — save periodically and probe sweep.

**Context vs. target encoder for evaluation.** The paper is explicit: *"We use the target-encoder for evaluation and average pool its output to produce a global image representation."* Both configs default to `encoder_source: target_encoder`. Late in training the two are nearly identical (EMA momentum → 1.0), but early checkpoints differ meaningfully. 

**Per-block prediction.** Each target block is predicted independently in a single forward pass. The context is replicated once per block via `repeat_interleave`, so the batch dimension becomes `B × num_blocks` with each row holding the same context paired with exactly one block's mask tokens.

---

## Results

All results are from-scratch pretraining on a single RTX 4090 (24 GB). The linear probe trains **only** the final linear layer on frozen encoder features — no fine-tuning.

### STL-10 linear probe

| Model | Encoder params | Pretrain steps | Probe accuracy |
|---|---|---|---|
| Small | 4M | 30,000 | ~48% (plateau) |
| Medium (ViT-Base) | 114M | 100,000 | ~48% (plateau) |

Random chance is 10%; a fully supervised CNN reaches ~85%. The plateau at ~48% for both model sizes is the **data-limited regime**: 100k unlabelled images with no augmentations saturate a 4M encoder. The 114M ViT-B learns the same information from the same data — more capacity does nothing

**Key observations from training runs:**

- The best probe checkpoint is rarely the last: on the Base model, accuracy peaked at step 50,000 (0.483) and drifted to 0.468 at step 100,000.
- Probe weight decay is harmful — set `weight_decay: 0.0` for the linear head (it only has 2,570 / 769,000 parameters over STL / ImageNet).
- STL-10 pretraining throughput is ~70k patches/sec (GPU-bound on a small encoder); ImageNet throughput depends on JPEG decode parallelism from DataLoader workers.

---

## References

- [Assran et al. — Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture, CVPR 2023](https://arxiv.org/abs/2301.08243)
- [Official I-JEPA implementation (facebookresearch/ijepa)](https://github.com/facebookresearch/ijepa)
- [STL-10 dataset (Coates et al.)](https://cs.stanford.edu/~acoates/stl10/)
- [evanarlian/imagenet_1k_resized_256 on HuggingFace](https://huggingface.co/datasets/evanarlian/imagenet_1k_resized_256) — non-gated ImageNet mirror at 256×256
