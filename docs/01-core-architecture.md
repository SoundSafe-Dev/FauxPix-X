# Part 1: The Core Architecture (Full-Frame & Temporal Focus)

## Executive Summary

Abandon traditional face-cropping preprocessing. FauxPix adopts a framework inspired by **UNITE (Universal Network for Identifying Tampered and synthEtic videos)** combined with optical flow tracking to analyze the **entire frame** and the **physics of motion**.

---

## 1. The Spatial Backbone: SigLIP-So400M

### Rationale

Traditional deepfake detectors rely on face-cropping preprocessing, which fails when:
- The video contains no human faces (T2V synthetic scenes)
- The manipulation occurs in the background (environmental tampering)
- The subject is non-human (animals, objects, abstract scenes)

**SigLIP-So400M** serves as the foundation model to extract **domain-agnostic features** across the entire video frame.

### Implementation Requirements

```python
# Pseudo-code for backbone integration
from transformers import AutoModel, AutoProcessor

class SpatialBackbone(nn.Module):
    def __init__(self):
        self.siglip = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
        self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    
    def forward(self, frames):
        # frames: [B, T, C, H, W] -> Process each frame through SigLIP
        # Returns: [B, T, D] feature embeddings
        pass
```

### Key Specifications

| Attribute | Value |
|-----------|-------|
| Model | SigLIP-So400M |
| Input Resolution | 384x384 |
| Patch Size | 14x14 |
| Feature Dimension | 1152 |
| Pre-trained Weights | Frozen initially, fine-tuned in SOTA phase |

---

## 2. Attention-Diversity (AD) Loss

### The Problem

Standard models over-focus on human faces due to:
- Dataset bias (majority of training clips contain faces)
- Saliency bias (faces are naturally attention-grabbing)
- Legacy architecture constraints (face-cropping pipelines)

This causes failure on fully synthetic T2V generations where the anomaly is in the environment, not the face.

### The Solution: AD Loss

AD Loss **forces the network to distribute its spatial attention** across diverse background elements of the video frame.

### Mathematical Formulation

```
AD_Loss = λ * (1 - Entropy(Attention_Map)) + (1 - λ) * Spatial_Dispersion_Penalty
```

Where:
- **Entropy(Attention_Map)**: Measures attention distribution diversity (higher = more diverse)
- **Spatial_Dispersion_Penalty**: Penalizes concentration in facial regions
- **λ**: Balancing hyperparameter (recommended: 0.7)

### Implementation Requirements

```python
class AttentionDiversityLoss(nn.Module):
    def __init__(self, lambda_weight=0.7, face_mask_weight=0.3):
        self.lambda_weight = lambda_weight
        self.face_mask_weight = face_mask_weight
    
    def forward(self, attention_maps, face_masks=None):
        # attention_maps: [B, H, W] spatial attention per frame
        # face_masks: [B, H, W] binary mask of facial regions
        
        # 1. Compute entropy of attention distribution
        entropy = -torch.sum(attention_maps * torch.log(attention_maps + 1e-8), dim=[1,2])
        
        # 2. Compute facial concentration penalty
        if face_masks is not None:
            face_attention = torch.sum(attention_maps * face_masks, dim=[1,2])
            spatial_penalty = face_attention  # Penalize high face attention
        else:
            spatial_penalty = 0
        
        loss = (self.lambda_weight * (1 - entropy.mean()) + 
                (1 - self.lambda_weight) * spatial_penalty.mean())
        
        return loss
```

### Expected Outcomes

| Metric | Without AD Loss | With AD Loss |
|--------|-----------------|--------------|
| Face Attention % | 65-75% | 25-35% |
| Background Attention % | 25-35% | 65-75% |
| T2V Synthetic Detection | Poor | Strong |
| Environmental Anomaly Detection | Fails | Detects |

---

## 3. Optical Flow Temporal Stream

### The Problem

AI video generators struggle with **temporal consistency**:
- Flickering textures across frames
- Physics-defying movement (unnatural acceleration)
- Inconsistent lighting/shadows over time
- Non-coherent motion in background elements

### The Solution: GC-ConsFlow-Inspired Optical Flow

Implement an **optical flow consistency stream** that extracts motion-compensated optical flow residuals between frames.

### Architecture

```
Frame_t          Frame_t+1
   |                 |
   v                 v
[Feature Extractor] [Feature Extractor]
   |                 |
   +--------v--------+
            |
    [Optical Flow Network]
            |
    +-------v-------+
    |               |
[Flow Field]    [Flow Residuals]
    |               |
    +-------v-------+
            |
    [Temporal Consistency Module]
            |
    [Anomaly Score]
```

### Implementation Requirements

```python
class OpticalFlowStream(nn.Module):
    def __init__(self, flow_backbone='raft'):
        self.flow_extractor = RAFT()  # or PWC-Net, FlowFormer
        self.motion_encoder = TemporalEncoder()
        self.consistency_scorer = ConsistencyMLP()
    
    def forward(self, frame_t, frame_t_plus_1):
        # 1. Extract optical flow
        flow_field = self.flow_extractor(frame_t, frame_t_plus_1)
        
        # 2. Compute motion-compensated residuals
        warped_t_plus_1 = warp_frame(frame_t_plus_1, flow_field)
        residuals = frame_t - warped_t_plus_1
        
        # 3. Encode temporal features
        temporal_features = self.motion_encoder(residuals, flow_field)
        
        # 4. Score consistency
        consistency_score = self.consistency_scorer(temporal_features)
        
        return consistency_score
```

### Key Optical Flow Metrics to Extract

| Metric | Description | Anomaly Indicator |
|--------|-------------|-------------------|
| Flow Magnitude Variance | Variation in motion strength | High variance = flickering |
| Flow Direction Consistency | Coherence of motion vectors | Low coherence = chaotic motion |
| Residual Energy | Pixel difference after compensation | High energy = physics violation |
| Temporal Gradient | Rate of flow change | Unnatural acceleration |

---

## 4. Full Architecture Integration

### Two-Stream Design

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Video                          │
│                    [B, T, C, H, W]                          │
└──────────────────┬──────────────────────┬───────────────────┘
                   |                      |
       ┌───────────v──────────┐  ┌────────v────────┐
       |   SPATIAL STREAM     |  | TEMPORAL STREAM |
       |                      |  |                 |
       |  SigLIP-So400M       |  | Optical Flow    |
       |  + AD Loss           |  | + Consistency   |
       |                      |  |                 |
       |  Output: [B, D_s]    |  | Output: [B, D_t]|
       └───────────┬──────────┘  └────────┬────────┘
                   |                      |
                   +----------v-----------+
                              |
                    [Fusion Module]
                              |
                    [Classification Head]
                              |
                       [Real / Fake]
```

### Fusion Strategy

```python
class FauxPixArchitecture(nn.Module):
    def __init__(self):
        self.spatial_backbone = SpatialBackbone()
        self.temporal_stream = OpticalFlowStream()
        self.fusion = CrossAttentionFusion(d_s=1152, d_t=512)
        self.classifier = DeepfakeClassifier(input_dim=1664)
    
    def forward(self, video_clip):
        # video_clip: [B, T, C, H, W]
        
        # Spatial features with attention maps for AD Loss
        spatial_feats, attention_maps = self.spatial_backbone(video_clip)
        
        # Temporal features from optical flow
        temporal_feats = self.temporal_stream(video_clip)
        
        # Cross-modal fusion
        fused = self.fusion(spatial_feats, temporal_feats)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits, attention_maps
```

---

## 5. Training Objectives

### Combined Loss Function

```python
Total_Loss = BCE_Loss + α * AD_Loss + β * Temporal_Consistency_Loss
```

| Loss Component | Weight (α/β) | Description |
|----------------|--------------|-------------|
| BCE_Loss | 1.0 | Standard binary cross-entropy |
| AD_Loss | 0.3 | Attention diversity penalty |
| Temporal_Consistency_Loss | 0.2 | Flow coherence regularization |

---

## 6. Inference Pipeline

### Frame Sampling Strategy

For a video of length N frames:
1. Sample T frames uniformly (e.g., T=16 for short clips, T=32 for long)
2. Extract optical flow between consecutive sampled pairs
3. Run forward pass through both streams
4. Aggregate scores across temporal windows
5. Output: Real/Fake probability + Confidence score

### Latency Targets

| Phase | Target Latency | Hardware |
|-------|---------------|----------|
| Commercial GTM | < 2 sec/video | 1x A100 |
| SOTA | < 5 sec/video | 8x A100 (batched) |

---

## 7. Key Implementation Notes for Dev Team

### Must-Implement Components

1. **SigLIP Integration**: Use `transformers` library, handle 384x384 resizing
2. **AD Loss Module**: Implement entropy-based attention diversity with face masking
3. **Optical Flow Extraction**: RAFT recommended for speed/accuracy trade-off
4. **Two-Stream Fusion**: Cross-attention or simple concatenation + MLP

### Critical Design Decisions

| Decision | Recommendation | Rationale |
|----------|----------------|-----------|
| Frame sampling | Uniform at 4-8 FPS | Balances temporal coverage and compute |
| Optical flow pre-computation | Yes for SOTA, No for Commercial | SOTA has 3M videos requiring I/O optimization |
| Face detection | Use only for AD Loss masking | Not for cropping - maintain full-frame analysis |
| Batch size | Maximize based on VRAM | 80GB A100 allows larger batches with full frames |

---

## References

- UNITE: Universal Network for Identifying Tampered and synthEtic videos
- SigLIP: Sigmoid Loss for Language Image Pre-Training
- RAFT: Recurrent All-Pairs Field Transforms for Optical Flow
- GC-ConsFlow: Geometry-Consistent Optical Flow for Video Consistency
