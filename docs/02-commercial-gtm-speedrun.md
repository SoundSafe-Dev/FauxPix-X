# Part 2: The Commercial Go-To-Market (GTM) Speedrun

## Executive Summary

Train a **lightweight, highly efficient model** capable of catching modern Sora/Runway generations for enterprise clients without requiring massive compute overhead. This is the MVP to get to market quickly and start generating revenue while the SOTA model trains.

**Timeline Target**: 4-6 weeks from data ingestion to production API

---

## 1. The Mindset

### Commercial Priorities (Ranked)

1. **Inference Speed**: Enterprise clients need < 2 second turnaround
2. **Memory Stability**: Consistent RAM usage for predictable scaling
3. **Modern Generator Detection**: Must catch Sora, RunwayML, Pika Labs, HeyGen
4. **Deployment Simplicity**: Single container, minimal dependencies
5. **Accuracy Floor**: > 90% on VID-AID validation set

### Trade-offs Accepted

| What We Sacrifice | What We Gain |
|-------------------|--------------|
| SOTA benchmark scores | Time to market |
| Training on 3M+ clips | Stable 14K curated dataset |
| Multi-modal fusion (audio) | Simpler deployment |
| Extreme resolution support | Consistent performance |

---

## 2. Compute Provisioning

### Hardware Specification

| Component | Specification | Purpose |
|-----------|-------------|---------|
| GPU | 1x NVIDIA A100 (80GB VRAM) | Model training and inference |
| CPU | 16+ cores | Data preprocessing, optical flow extraction |
| RAM | 128GB+ | Dataset caching, video decoding |
| Storage | 1TB Local NVMe SSD | Dataset storage, model checkpoints |

### Why 80GB VRAM is Mandatory

Video processing requires sampling multiple frames per clip:
- Batch size 8 with 16 frames @ 384x384 = ~24GB VRAM
- Optical flow computation overhead = ~16GB VRAM
- Gradient accumulation + optimizer states = ~20GB VRAM
- **Total**: ~60GB → 80GB provides necessary headroom for stable training

### Alternative Configurations

If A100 unavailable, acceptable fallbacks:
- 2x A6000 (48GB each) with model parallelism
- 1x H100 (80GB) - superior but more expensive
- **Not acceptable**: Anything < 48GB VRAM (batch size collapse)

---

## 3. The Commercial Dataset

### Target: ~14,100 Clips

Focus strictly on **quality T2V samples** using PyTorch `SubsetRandomSampler` for efficient epoch management.

### Dataset Composition

```
Total: 14,100 clips
├── AI-Generated: 10,100 clips (71.6%)
│   ├── VID-AID: 10,000 clips (9 T2V models)
│   └── RealGuard-2025: 100 clips (Sora, Runway, Pika, HeyGen)
└── Real: 4,000 clips (28.4%)
    └── VID-AID: 4,000 clips (authentic videos)
```

### Primary Dataset: VID-AID

| Attribute | Specification |
|-----------|---------------|
| Total Clips | 14,000 (10,000 AI + 4,000 Real) |
| AI Models | 9 different modern T2V generators |
| Duration | 5-15 seconds per clip |
| Resolution | 720p minimum |
| Annotation | Binary (Real/Fake) |

**VID-AID Model Distribution (AI clips)**:

| Generator | Count | Notes |
|-----------|-------|-------|
| ModelScope | 1,500 | Alibaba's T2V model |
| VideoCrafter2 | 1,500 | Open-source T2V |
| Pika Labs | 1,200 | Commercial platform |
| RunwayML Gen-2 | 1,200 | Industry leader |
| Stable Video Diffusion | 1,200 | Stability AI |
| AnimateDiff | 1,000 | Motion adapter |
| LaVie | 1,000 | High-res T2V |
| I2VGen-XL | 1,000 | Image-to-video |
| Open-Sora | 400 | Open-source Sora-like |

### Secondary Dataset: RealGuard-2025

| Attribute | Specification |
|-----------|---------------|
| Total Clips | 100 multimedia samples |
| Focus | Cutting-edge generators |
| Sora | 25 clips (if available) |
| RunwayML | 25 clips (latest Gen-3 if available) |
| Pika Labs 1.5 | 25 clips |
| HeyGen | 25 clips (avatar/video generation) |
| Resolution | 1080p where available |
| Purpose | Validation against unseen, latest generators |

### DataLoader Configuration

```python
from torch.utils.data import DataLoader, SubsetRandomSampler

# Commercial GTM DataLoader
class CommercialDatasetConfig:
    batch_size = 8  # Fits in 80GB VRAM with 16 frames
    num_workers = 8
    frame_count = 16  # Frames sampled per video
    frame_size = 384  # SigLIP input size
    
    # Use SubsetRandomSampler for efficient sampling
    sampler = SubsetRandomSampler(
        indices=list(range(total_clips)),
        num_samples=epoch_size  # Control samples per epoch
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
```

---

## 4. Training Regimen

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 50-75 | Early stopping on validation plateau |
| Learning Rate | 1e-4 with cosine decay | Stable convergence |
| Optimizer | AdamW | Standard for vision transformers |
| Weight Decay | 0.01 | Prevent overfitting on 14K clips |
| Frame Sampling | 16 frames uniform | Balance coverage and compute |
| AD Loss Weight | 0.3 | From architecture spec |
| Temporal Loss Weight | 0.2 | From architecture spec |

### Training Loop Structure

```python
# Commercial training loop (single A100)
def train_commercial_epoch(model, dataloader, optimizer):
    for batch in dataloader:
        videos, labels = batch  # [B, T, C, H, W]
        
        # Forward pass
        logits, attention_maps = model(videos)
        
        # Compute losses
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels)
        ad_loss = attention_diversity_loss(attention_maps)
        temporal_loss = temporal_consistency_loss(videos, logits)
        
        total_loss = bce_loss + 0.3 * ad_loss + 0.2 * temporal_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

### Validation Strategy

| Validation Split | Purpose | Frequency |
|------------------|---------|-----------|
| VID-AID Holdout (10%) | In-domain accuracy | Every epoch |
| RealGuard-2025 | Unseen generator detection | Every 5 epochs |
| Self-collected Real Videos | False positive rate | Weekly manual check |

---

## 5. Model Packaging

### Checkpoint Deliverables

```
commercial_gtm_v1/
├── model.pt                    # PyTorch model weights
├── config.yaml                 # Architecture hyperparameters
├── preprocessor.pkl            # Frame sampling config
├── inference.py                # Standalone inference script
├── requirements.txt            # Dependencies
└── README.md                   # API documentation
```

### Inference API Specification

```python
# Expected inference interface
class FauxPixCommercialAPI:
    def __init__(self, model_path, device='cuda'):
        self.model = load_model(model_path)
        self.device = device
    
    def predict(self, video_path):
        """
        Args:
            video_path: Path to video file
        
        Returns:
            {
                'is_fake': bool,
                'confidence': float,  # 0.0 - 1.0
                'processing_time_ms': int,
                'frames_analyzed': int
            }
        """
        pass
```

### Performance Benchmarks

| Metric | Target | Minimum Acceptable |
|--------|--------|-------------------|
| Accuracy (VID-AID) | > 92% | > 90% |
| Accuracy (RealGuard) | > 85% | > 80% |
| Inference Time | < 2s | < 5s |
| False Positive Rate | < 5% | < 10% |
| False Negative Rate | < 8% | < 12% |

---

## 6. Deployment Checklist

### Pre-Deployment

- [ ] Model validated on 100% of RealGuard-2025
- [ ] Inference time benchmarked on A100
- [ ] Docker container tested end-to-end
- [ ] API documentation complete
- [ ] Monitoring/logging configured

### Production API Endpoints

```
POST /api/v1/detect
Request: { "video_url": "https://..." }
Response: { "is_fake": true, "confidence": 0.94, "latency_ms": 1200 }

POST /api/v1/detect/batch
Request: { "video_urls": ["...", "..."] }
Response: { "results": [...], "batch_latency_ms": 5000 }

GET /api/v1/health
Response: { "status": "healthy", "model_version": "commercial_gtm_v1" }
```

---

## 7. Timeline

| Week | Milestone |
|------|-----------|
| 1 | Data ingestion pipeline, VID-AID preprocessing |
| 2 | Architecture implementation, initial training |
| 3 | Hyperparameter tuning, validation on RealGuard |
| 4 | Optimization, inference API development |
| 5 | Testing, documentation, deployment prep |
| 6 | Production deployment, monitoring setup |

---

## 8. Success Criteria

The Commercial GTM Speedrun is **complete** when:

1. Model achieves > 90% accuracy on VID-AID validation
2. Model achieves > 80% accuracy on RealGuard-2025
3. Inference API responds in < 2 seconds per video
4. Docker container runs successfully on single A100
5. First enterprise client onboarded

---

## 9. Handoff to SOTA Phase

Once commercial model is deployed:

1. **Freeze commercial weights** → backup as baseline
2. **Scale compute** → provision 8x A100 cluster
3. **Expand dataset** → begin AV-Deepfake1M++ ingestion
4. **Unfreeze backbone** → enable SigLIP fine-tuning
5. **Pre-compute optical flow** → offline extraction for 3M videos

See [03-sota-brute-force-run.md](./03-sota-brute-force-run.md) for SOTA phase details.
