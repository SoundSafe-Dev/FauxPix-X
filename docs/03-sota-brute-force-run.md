# Part 3: The SOTA Brute-Force Run

## Executive Summary

Once the Commercial GTM model is shipped, scale up to build an **apex-tier foundation model** capable of leading video detection benchmarks. This phase leverages massive compute and data to push the state-of-the-art.

**Timeline Target**: 3-4 months from cluster provisioning to benchmark submission

---

## 1. The Mindset

### SOTA Priorities (Ranked)

1. **Benchmark Leadership**: Top-3 on AV-Deepfake1M++, FaceForensics++, Celeb-DF
2. **Generalization**: Robust performance across all known generators
3. **Segment-Level Precision**: Detect partial manipulations within clips
4. **Audio-Video Fusion**: Multi-modal detection capability
5. **Research Impact**: Publishable results, citation-worthy methodology

### Trade-offs Accepted

| What We Sacrifice | What We Gain |
|-------------------|--------------|
| Training speed | Maximum accuracy |
| Single-GPU deployment | Multi-GPU foundation model capability |
| Inference latency | Detection precision |
| Simple architecture | Complex multi-modal fusion |

---

## 2. Compute Provisioning

### Hardware Specification

| Component | Specification | Purpose |
|-----------|-------------|---------|
| GPU | 8x NVIDIA A100 (80GB VRAM) | Distributed training |
| Interconnect | NVLink + InfiniBand | Cross-GPU communication |
| CPU | 64+ cores | Data loading, preprocessing |
| RAM | 512GB+ | Large batch buffering |
| Storage | 2TB+ Local NVMe | Dataset + checkpoints + flow cache |
| Network | 10Gbps+ | Distributed dataset access |

### Why 8x A100

| Training Phase | GPUs Required | VRAM per GPU | Reason |
|--------------|---------------|--------------|--------|
| Backbone fine-tuning | 8 | 80GB | Full SigLIP unfreezing requires memory |
| Large batch training | 8 | 80GB | Global batch size 64+ for stable gradients |
| Multi-resolution | 8 | 80GB | Variable input sizes need padding |
| Audio-Video fusion | 8 | 80GB | Additional modality = memory overhead |

### Distributed Training Strategy

```python
# PyTorch DistributedDataParallel setup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_sota_training():
    # Initialize process group across 8 GPUs
    dist.init_process_group(backend='nccl')
    
    # Model with DDP wrapper
    model = FauxPixSOTA().cuda()
    model = DDP(model, device_ids=[local_rank], 
                output_device=local_rank,
                find_unused_parameters=True)  # For AD loss attention maps
    
    # Gradient scaling for mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    return model, scaler
```

---

## 3. The SOTA Dataset

### Target: ~3.1 Million+ Clips

Configure dataloaders for **lazy-loading** these exact sources:

```
Total: 3,120,000+ clips
├── AV-Deepfake1M++: 2,000,000 clips (64.1%)
│   └── Audio-visual deepfakes with segment-level annotations
├── OpenVid-1M: 1,000,000 clips (32.1%)
│   └── High-quality authentic baseline
└── GenVidDet: 120,000 clips (3.8%)
    └── Sub-sampled from 1.12M AI-generated collection
```

### Primary Dataset: AV-Deepfake1M++

| Attribute | Specification |
|-----------|---------------|
| Total Clips | 2,000,000+ |
| Annotations | Segment-level (partial manipulations) |
| Modalities | Audio + Video |
| Fake Types | Face-swap, lip-sync, full synthesis |
| Resolution | Mixed (360p to 1080p) |

**Annotation Granularity**:

```python
# AV-Deepfake1M++ annotation structure
sample_annotation = {
    "video_id": "vid_001",
    "is_fake": True,
    "segments": [
        {
            "start_frame": 0,
            "end_frame": 45,
            "fake_type": "face_swap",
            "confidence": 0.95
        },
        {
            "start_frame": 46,
            "end_frame": 120,
            "fake_type": "real",
            "confidence": 1.0
        }
    ],
    "audio_fake": False,
    "audio_visual_alignment": 0.92
}
```

### Baseline Dataset: OpenVid-1M

| Attribute | Specification |
|-----------|---------------|
| Total Clips | 1,000,000 |
| Content | High-quality authentic videos |
| Source | YouTube, Vimeo (licensed) |
| Duration | 10-60 seconds |
| Resolution | 720p minimum |
| Diversity | 500+ content categories |

**Purpose**: Establish robust "bona fide" baseline to prevent false positives.

### AI-Generated Dataset: GenVidDet (Subset)

| Attribute | Specification |
|-----------|---------------|
| Source Collection | 1,120,000 clips |
| Subsample Target | 120,000 clips (10%) |
| Generators | Pika, ModelScope, VideoCraft2, Sora, others |
| Selection Strategy | Stratified sampling across generators |
| Quality Filter | Minimum 720p, 5+ seconds |

**Stratification Distribution**:

| Generator | Subsample Count | Notes |
|-----------|-----------------|-------|
| Pika Labs | 25,000 | Latest generations |
| ModelScope | 20,000 | Open T2V diversity |
| VideoCraft2 | 20,000 | High-quality synthesis |
| Sora (if leaked/public) | 15,000 | Priority capture |
| RunwayML Gen-3 | 15,000 | Commercial cutting-edge |
| HeyGen | 10,000 | Avatar generation |
| Stable Video Diffusion | 10,000 | Open source |
| Luma Dream Machine | 5,000 | Emerging platform |

---

## 4. DataLoader Optimization

### The I/O Bottleneck Problem

Extracting optical flow residuals across 3 million videos creates a **massive I/O bottleneck**.

**Without Optimization**:
- Raw video loading: ~50ms per clip
- Optical flow extraction: ~200ms per clip
- Total per batch (64 clips): ~16 seconds
- Training stalls on I/O 80% of time

### Solution: Pre-compute Optical Flow Offline

```python
# Offline optical flow pre-computation
class OpticalFlowCache:
    def __init__(self, output_dir='flow_cache/'):
        self.output_dir = output_dir
    
    def precompute_and_cache(self, video_dataset):
        """Run once before training"""
        for video_path, video_id in video_dataset:
            # Extract flow
            flows = self.extract_flow_sequence(video_path)
            
            # Save as tensor
            cache_path = f"{self.output_dir}/{video_id}.pt"
            torch.save(flows, cache_path)
    
    def load_cached_flow(self, video_id):
        """Fast load during training"""
        return torch.load(f"{self.output_dir}/{video_id}.pt")
```

### Optimized DataLoader Configuration

```python
class SOTADataLoaderConfig:
    # Batch configuration for 8x A100
    global_batch_size = 64  # 8 per GPU
    num_workers_per_gpu = 4
    
    # Pre-computed flow caching
    use_cached_flow = True
    flow_cache_dir = "/mnt/nvme/flow_cache/"
    
    # Lazy loading with memory mapping
    lazy_load = True
    memory_map = True
    
    # Prefetching
    prefetch_factor = 4
    persistent_workers = True
    
    # Pin memory for faster GPU transfer
    pin_memory = True
    non_blocking = True

# Dataloader with flow caching
class SOTADataset(Dataset):
    def __init__(self, video_list, flow_cache_dir):
        self.video_list = video_list
        self.flow_cache_dir = flow_cache_dir
    
    def __getitem__(self, idx):
        video_path = self.video_list[idx]
        video_id = extract_id(video_path)
        
        # Load frames (lazy)
        frames = load_video_frames(video_path, num_frames=32)
        
        # Load pre-computed flow (fast)
        flows = torch.load(f"{self.flow_cache_dir}/{video_id}.pt")
        
        label = self.get_label(video_id)
        
        return frames, flows, label
```

### Cache Storage Requirements

| Data | Size per Clip | 3.1M Clips | Storage Strategy |
|------|---------------|------------|------------------|
| Video frames (raw) | 50MB | 155TB | Lazy load from network storage |
| Optical flow cache | 5MB | 15.5TB | Local NVMe (2TB) + tiered storage |
| Audio spectrograms | 2MB | 6.2TB | Local NVMe or RAM disk |

**Flow Cache Pre-computation Time**: ~72 hours on 8x A100 cluster (run once)

---

## 5. Training Regimen

### Phase 1: Warmup (Epochs 1-10)

| Parameter | Value |
|-----------|-------|
| Backbone | Frozen (SigLIP) |
| Trainable | Temporal stream, fusion, classifier |
| Learning Rate | 1e-3 |
| Batch Size | 64 |
| Loss Weights | BCE=1.0, AD=0.3, Temporal=0.2 |

### Phase 2: Full Training (Epochs 11-100)

| Parameter | Value |
|-----------|-------|
| Backbone | **Unfrozen** (end-to-end) |
| Learning Rate | 1e-4 with cosine decay |
| Batch Size | 64 |
| Gradient Clipping | 1.0 (stabilize SigLIP fine-tuning) |
| Mixed Precision | bf16 (faster than fp32, stable) |

### Phase 3: Fine-tuning (Epochs 101-150)

| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-5 |
| Data Augmentation | Heavy (color jitter, Gaussian blur, frame dropout) |
| Focus | Hard negative mining |

### Distributed Training Loop

```python
def train_sota_epoch(model, dataloader, optimizer, scaler, epoch):
    model.train()
    
    for batch_idx, (frames, flows, labels) in enumerate(dataloader):
        frames = frames.cuda(non_blocking=True)
        flows = flows.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits, attention_maps = model(frames, flows)
            
            bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
            ad_loss = attention_diversity_loss(attention_maps)
            temporal_loss = temporal_consistency_loss(flows, logits)
            
            total_loss = bce_loss + 0.3 * ad_loss + 0.2 * temporal_loss
        
        # Scaled backward
        scaler.scale(total_loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Logging
        if batch_idx % 100 == 0 and dist.get_rank() == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {total_loss.item():.4f}")
```

---

## 6. Validation & Benchmarking

### Validation Splits

| Dataset | Split | Frequency | Purpose |
|---------|-------|-----------|---------|
| AV-Deepfake1M++ | 50K holdout | Every epoch | Primary accuracy metric |
| FaceForensics++ | Full | Weekly | Cross-dataset generalization |
| Celeb-DF | Full | Weekly | Celeb-specific performance |
| DFDC | Full | Monthly | Industry benchmark |
| RealGuard-2025 | Full | Every epoch | Latest generator detection |

### Target Benchmark Scores

| Benchmark | Current SOTA | FauxPix Target |
|-----------|--------------|----------------|
| AV-Deepfake1M++ (AUC) | 0.94 | **0.96** |
| FaceForensics++ (AUC) | 0.99 | **0.995** |
| Celeb-DF (AUC) | 0.97 | **0.98** |
| DFDC (AUC) | 0.85 | **0.88** |

---

## 7. Model Checkpointing

### Checkpoint Strategy

```python
# Save every 10 epochs + best model
checkpoint_config = {
    'save_frequency': 10,  # epochs
    'keep_last_n': 5,
    'save_optimizer': True,
    'save_scaler': True,
    'compress': False  # Full precision for reproducibility
}

# Checkpoint structure
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.module.state_dict(),  # Unwrap DDP
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'best_auc': best_auc,
    'config': model_config,
}
```

### Final Model Deliverables

```
sota_fauxpix_v1/
├── model.pt                    # 1.2GB+ (full foundation model)
├── config.yaml                 # Complete hyperparameters
├── training_log.jsonl          # Per-epoch metrics
├── benchmark_results.json      # Final benchmark scores
├── inference.py                # Optimized inference script
├── requirements.txt            # Exact dependency versions
├── flow_precompute.py          # Cache generation script
├── LICENSE                     # Model license
└── MODEL_CARD.md              # Ethics, limitations, intended use
```

---

## 8. Timeline

| Month | Milestone |
|-------|-----------|
| 1 | Cluster setup, data ingestion, flow cache pre-computation |
| 2 | Phase 1-2 training, validation protocol development |
| 3 | Phase 3 fine-tuning, benchmark submission preparation |
| 4 | Final evaluation, paper writing, model release |

---

## 9. Execution Rules Summary

1. **Pre-compute optical flow offline** → 15TB cache, 72 hours prep time
2. **Use DistributedDataParallel** → 8x A100 cluster utilization
3. **Enable mixed precision (bf16)** → 2x speedup, stable training
4. **Unfreeze SigLIP in Phase 2** → End-to-end optimization
5. **Validate on multiple benchmarks** → Generalization verification
6. **Keep last 5 checkpoints** → Recovery from divergence
7. **Log everything** → Reproducibility and paper evidence

---

## 10. Success Criteria

The SOTA Brute-Force Run is **complete** when:

1. Model achieves > 0.96 AUC on AV-Deepfake1M++
2. Model achieves > 0.98 AUC on FaceForensics++
3. Benchmark paper submitted to CVPR/NeurIPS/ICCV
4. Model checkpoint reproducible from training logs
5. Inference API demonstrates < 5s latency per video (batched)
6. Model card documents limitations and ethical use

---

## References

- AV-Deepfake1M++: Large-scale audio-visual deepfake dataset
- OpenVid-1M: Million-scale authentic video dataset
- GenVidDet: Large-scale generated video detection dataset
- PyTorch Distributed: Best practices for multi-GPU training
