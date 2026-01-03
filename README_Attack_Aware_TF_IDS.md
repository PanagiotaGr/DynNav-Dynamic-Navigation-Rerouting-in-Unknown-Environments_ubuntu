
# Attack-Aware TF Integrity Monitoring with CUSUM 

## Overview
This work documents the design and implementation of a **stealth drift detection mechanism** for ROS 2 TF streams,
motivated by cybersecurity threats against localization and navigation pipelines.

We focus on **TF spoofing attacks** that introduce *small but persistent drift* in pose estimates, which remain within
physical bounds and therefore bypass simple rule-based or threshold-based detectors.

To address this, we integrate **sequential change detection (CUSUM)** into a TF integrity monitor, inspired by
innovation-based IDS methods used in state estimation (e.g., NIS-based detectors for Kalman filters).

---

## Threat Model
We consider an attacker that:
- Publishes a forged TF transform (`odom → base_link_spoofed`)
- Injects **low-magnitude translational and rotational drift**
- Respects physical velocity limits to remain stealthy

Such attacks are realistic in ROS systems where TF is trusted implicitly by planners and costmaps.

---

## Baseline Limitation
A baseline TF integrity monitor checks:
- Linear velocity: \( v \le v_{max} \)
- Angular velocity: \( \omega \le \omega_{max} \)

This approach fails against **slow drift**, as observed experimentally:
- Typical drift: ~0.02 m/s, ~0.01 rad/s
- Normalized ratios: \( v/v_{max} \approx 0.1 \), \( \omega/\omega_{max} \approx 0.03 \)
- No instantaneous violation occurs

---

## Proposed Method: CUSUM-based TF IDS

### Per-step Score
At each TF update, we compute:

\[
\text{score}_t = \max \left( \frac{v_t}{v_{max}}, \frac{|\omega_t|}{\omega_{max}} \right)
\]

This produces a **normalized anomaly signal**.

### Sequential Detection
We apply a **one-sided CUSUM**:

\[
g_t = \max(0, g_{t-1} + (\text{score}_t - k))
\]

An alarm is raised when:

\[
g_t \ge h
\]

Where:
- \( k \) controls sensitivity to small deviations
- \( h \) controls detection delay vs false alarms

---

## Implementation
- Node: `tf_integrity_monitor_cusum.py`
- Subscribes implicitly to TF (`tf2_ros.Buffer`)
- Publishes:
  - `/ids/tf_score` — instantaneous normalized score
  - `/ids/tf_cusum` — accumulated CUSUM statistic
  - `/ids/tf_alarm` — Boolean alarm (latched)

The detector is **drop-in**, requires no map or ground truth, and operates purely on TF dynamics.

---

## Experimental Observation
Using a spoofed TF with slow drift:
- Instantaneous score remains below 1.0
- CUSUM accumulates steadily over time
- Alarm is triggered after several seconds (configurable)

This confirms that **sequential analysis detects attacks invisible to static checks**.

---

## Significance
This experiment demonstrates that:
- TF streams are a viable attack surface in ROS
- Physical-limit checks are insufficient for security
- Sequential IDS methods (CUSUM/EWMA) are effective and lightweight

The approach naturally extends to:
- Costmap consistency checks
- Planner deviation monitoring
- Multi-sensor trust and fusion security

---


