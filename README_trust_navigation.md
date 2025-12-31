# Human–Robot Trust Dynamics & Preference-Aware Risk Navigation

This repository implements a **human-aware, risk-sensitive navigation module** that couples:

1. **Robot self-trust**
2. **Estimated human trust in the robot**
3. **Human preference awareness**

into a unified **risk-aware navigation policy controller**.  
The goal is to transform a static planner into an **adaptive, self-aware, human-aligned navigation system**.

---

## 🔍 Concept Overview

### Baseline Risk-Aware Navigation

The underlying navigation stack optimizes a trajectory policy \(\pi\) using a risk-weighted cost function:

\[
\text{Cost}(\pi) = L(\pi) + \lambda \cdot R(\pi)
\]

- \(L(\pi)\): nominal objective (e.g., path length, time, control effort)  
- \(R(\pi)\): risk functional (e.g., collision risk, uncertainty, OOD / drift penalties)  
- \(\lambda\): risk weight — **high** \(\lambda\) → conservative behavior, **low** \(\lambda\) → aggressive behavior

The baseline system already includes:

- **Self-trust / model confidence**
- **Out-of-distribution (OOD) awareness**
- **Drift awareness**
- **Calibrated uncertainty estimates**

---

## 👤 Human Preference Layer

A human can express high-level navigation preferences such as:

- “Prefer safer route even if slower”
- “Reach fast, I accept risk”
- “Avoid dark / low-feature regions”
- “Balanced”

These are parsed into:

- A **continuous risk preference** \(h \in [0, 1]\)  
  - \(h \approx 0\): very conservative  
  - \(h \approx 1\): very aggressive / risk-tolerant
- **Semantic constraints**, e.g.:
  - `avoid_dark_areas`
  - `avoid_low_feature_areas`
  - `prefer_well_mapped`

The human preference influences the effective risk weight:

\[
\lambda_{\text{effective}} =
f(\lambda_{\text{robot}},\; h,\; \alpha_{\text{human}})
\]

where:

- \(\lambda_{\text{robot}}\) is the robot’s internally preferred risk weight  
- \(h\) is the human’s current risk preference  
- \(\alpha_{\text{human}} \in [0,1]\) is a **human influence scale** (how much the robot currently lets the human override its default)

`λ_effective` is what is ultimately sent to the planner.

---

## 🤝 Human–Robot Trust Dynamics

The module maintains **two coupled trust processes**, both normalized in \([0,1]\):

1. **Robot self-trust** \(T_{\text{robot}}\)
2. **Estimated human trust in the robot** \(T_{\text{human}}\)

Both evolve over time according to events such as successes, failures, and overrides.

### Robot Self-Trust

`T_robot` **increases** when:

- Navigation episodes **succeed** (goal reached without safety violations)

and **decreases** when:

- **Near-misses** occur (e.g., very small safety margins)
- **Failures** occur (e.g., collision, unrecoverable state)
- The **human overrides** the controller

### Human Trust in the Robot

`T_human` is an **internal estimate** maintained by the robot. It is updated using observable signals:

- **Increases** with:
  - Successful navigation episodes
  - Explicit **human approval** or lack of intervention

- **Decreases** with:
  - Failures / unsafe behavior
  - Frequent human overrides
  - Near-misses that indicate discomfort

Both trust variables are clipped to \([0,1]\).

---

## 🧠 Trust → Policy Mapping

Trust values are mapped into navigation behavior via \(\lambda\) and the **human influence scale**:

### 1. Robot Risk Weight \(\lambda_{\text{robot}}\)

- **Low** `T_robot` → **increase** \(\lambda\) (more conservative, safer)  
- **High** `T_robot` → **decrease** \(\lambda\) (more aggressive / performance-oriented)

### 2. Human Influence Scale \(\alpha_{\text{human}}\)

- **High** `T_human` → human preferences are trusted more  
- **Low** `T_human` → human preferences impact the policy less

Thus over time, the robot:

- Learns **how much it should trust itself**, and  
- Learns **how much it believes the human trusts it**, modulating the degree of human control.

### 3. Safe Mode

A **Safe Mode** is activated when any of the following holds:

- `T_robot` is below a threshold (robot has low confidence)
- `T_human` is below a threshold (human is estimated to distrust the system)

In Safe Mode, the controller:

- Enforces **high risk weight** \(\lambda\) (very conservative)  
- May **restrict** high-risk maneuvers  
- Can **prioritize well-mapped, high-confidence regions**

---

## 📂 Repository Structure

The key entry-point scripts are:

- `run_trust_dynamics_demo.py`  
  Demo focusing on **trust dynamics only** (no human preference integration).

- `run_trust_and_preferences_demo.py`  
  Demo of **full integration**: trust + human preference + effective risk update.

- `save_trust_preference_results.py`  
  Exports logged runs into a CSV file (`trust_preference_results.csv`) for external analysis.

Your actual repository may include additional modules such as:

- `trust_model/` – trust update rules, normalization, and thresholds  
- `navigation_core/` – baseline planner, cost functions, risk computation  
- `human_interface/` – parsing of human preference text to (h, semantic constraints)  
- `config/` – YAML / JSON configuration for parameters and thresholds

(Names may differ; adjust this README to match your actual folder structure.)

---

## ⚙️ Installation

```bash
git clone <YOUR_REPO_URL>.git
cd <YOUR_REPO_NAME>
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
pip install -r requirements.txt  # if provided
```

Typical dependencies (for guidance only) may include:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `pyyaml` (or similar, if using config files)

Adjust this section according to your actual project.

---

## 🚀 Quick Start

### 1️⃣ Trust Dynamics Only

Runs a standalone demo of **robot self-trust** and **human trust** evolution.

```bash
python3 run_trust_dynamics_demo.py
```

This demo typically visualizes or logs, for each step:

- `self_trust_robot`
- `human_trust_in_robot`
- `lambda_robot`
- `human_influence_scale`
- `safe_mode` flag

Use this to verify that trust updates behave as expected under different event sequences.

---

### 2️⃣ Trust + Human Preferences Integration

Runs a complete loop where:

- An **event** occurs (`SUCCESS`, `FAILURE`, `HUMAN_OVERRIDE`, etc.)
- A **human preference** is provided (text)
- Trust states are updated
- Risk weights and human influence are recomputed
- The final `λ_effective` is sent to the planner

```bash
python3 run_trust_and_preferences_demo.py
```

A typical step will log:

- `event`
- `human_preference_text`
- `self_trust_robot`
- `human_trust_in_robot`
- `lambda_robot`
- `human_influence_scale`
- `human_risk_preference_h`
- `lambda_effective`
- `safe_mode`

This demo is ideal for **ablation studies** and **user studies** on how trust and preference co-evolve.

---

### 3️⃣ Export Results for Plots

To export results to CSV for offline analysis:

```bash
python3 save_trust_preference_results.py
```

This generates:

- `trust_preference_results.csv`

with columns such as:

- `step`
- `event`
- `human_preference_text`
- `self_trust_robot`
- `human_trust_in_robot`
- `lambda_robot`
- `lambda_effective`
- `human_risk_preference_h`
- `human_influence_scale`
- `safe_mode`

You can then analyze this file using:

- **Excel / LibreOffice**
- **pandas** (Python)
- **matplotlib / seaborn** for plotting
- Any other data analysis pipeline for research figures

---

## 📊 Research Use & Extensions

This framework provides a clean testbed for questions in:

- **Human–Robot Interaction (HRI)**
- **Trust-aware autonomy**
- **Risk-aware navigation**
- **Explainable decision-making**

Example research directions:

- How do **different trust-update rules** affect long-term behavior?
- How strongly should human preferences influence navigation under **low trust**?
- How do **failures, near-misses, and overrides** reshape risk profiles?
- What is the relationship between **subjective human comfort** and \(\lambda_{\text{effective}}\)?

---

## 🔮 Possible Next Steps

The codebase is designed to be extended towards:

- **ROS / ROS2 integration** with real robots  
- Real-world navigation experiments with **online trust adaptation**  
- **Bayesian** or **reinforcement learning** approaches to learn trust dynamics  
- A GUI / web UI for real-time human preference input  
- Full, **paper-ready experimental pipelines** with reproducible scripts

---

## 📜 Citation (Placeholder)

If you use this repository in academic work, please consider citing it.  
(Replace the following with your actual citation.)

```text
@inproceedings{yourkey202Xtrustnav,
  title     = {Human--Robot Trust Dynamics and Preference-Aware Risk Navigation},
  author    = {Your Name and Collaborators},
  booktitle = {Proceedings of ...},
  year      = {202X}
}
```

---

## 📄 License

Specify your license here, e.g.:

- MIT
- BSD-3-Clause
- Apache-2.0

```text
Copyright (c) <Year>, <Your Name>

Licensed under the <Your License> License.
See the LICENSE file for details.
```

---

## ✅ Summary

This module delivers:

- **Human-aware, trust-evolving navigation**
- Adaptive **risk weighting** via \(\lambda_{\text{effective}}\)
- Integration of **semantic preferences** (e.g., avoid dark / low-feature regions)
- **Safe Mode** for low-trust situations
- Rich logging for **analysis and research**

It provides a practical bridge between **human-centered robotics**,  
**trust modeling**, and **risk-aware motion planning**.
