## Innovation-Based Intrusion Detection (IDS)

This module implements an innovation-based IDS embedded in UKF sensor fusion.
Attacks are injected at the measurement level (replay, bias, ramp, noise)
without modifying estimator logic.

### Features
- Mahalanobis-distance IDS with χ² thresholding
- Separation of detection and mitigation
- Adaptive trust scaling
- Per-timestep logging and reproducible experiments

### Reproducibility
Run:
```bash
python3 eval_ids_replay.py --mode replay
