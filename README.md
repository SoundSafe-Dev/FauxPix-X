# FauxPix Full-Frame Video Deepfake Detection

> **Universal video deepfake detection for the T2V/I2V era.** Detects synthetic content across the entire frame, not just faces.

## Overview

The video deepfake detection landscape has fundamentally changed. Legacy face-centric detectors fail against modern Text-to-Video (T2V) and Image-to-Video (I2V) models like Sora, RunwayML, and Pika Labs that generate entirely synthetic scenes, non-human subjects, and complex background modifications.

FauxPix adopts a **full-frame, physics-aware architecture** that analyzes the entire video frame and temporal motion consistency to detect AI-generated content regardless of subject type.

## Architecture Highlights

- **Spatial Backbone**: SigLIP-So400M for domain-agnostic feature extraction across full frames
- **Attention-Diversity (AD) Loss**: Forces distributed attention across background elements, catching environmental anomalies
- **Optical Flow Temporal Stream**: Motion-compensated optical flow residuals detect physics-defying temporal inconsistencies

## Documentation

| Document | Description |
|----------|-------------|
| [Core Architecture](./docs/01-core-architecture.md) | Full-Frame & Temporal Focus design (SigLIP, AD Loss, Optical Flow) |
| [Commercial GTM Speedrun](./docs/02-commercial-gtm-speedrun.md) | Lightweight enterprise model: 1x A100, VID-AID + RealGuard-2025 |
| [SOTA Brute-Force Run](./docs/03-sota-brute-force-run.md) | Apex foundation model: 8x A100, 3.1M+ clips from AV-Deepfake1M++, OpenVid-1M, GenVidDet |

## Quick Navigation

- **For Implementation Teams**: Start with [Core Architecture](./docs/01-core-architecture.md) to understand the technical foundation
- **For Product/Commercial Teams**: Review [Commercial GTM Speedrun](./docs/02-commercial-gtm-speedrun.md) for the enterprise MVP strategy
- **For Research Teams**: See [SOTA Brute-Force Run](./docs/03-sota-brute-force-run.md) for the full-scale training roadmap

## Project Structure

```
FauxPix/
├── docs/
│   ├── 01-core-architecture.md
│   ├── 02-commercial-gtm-speedrun.md
│   └── 03-sota-brute-force-run.md
├── src/                    # Implementation (TBD by dev team)
├── configs/                # Training configs (TBD by dev team)
├── data/                   # Dataset loaders (TBD by dev team)
└── README.md
```

## License

[To be determined by project stakeholders]
