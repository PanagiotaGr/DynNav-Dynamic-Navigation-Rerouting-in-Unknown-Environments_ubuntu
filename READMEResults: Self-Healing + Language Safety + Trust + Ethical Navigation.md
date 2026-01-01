# Results: Self-Healing + Language Safety + Trust + Ethical Navigation

This section presents the **simulation results** produced by the integrated framework:
- Self-Healing Navigation Policy  
- Language-Driven Safety Policy  
- Ethical Risk Layer  
- Trust Dynamics Layer  
- Adaptive Planner Modulation  

Although these results come from a **software simulation**, the logic is scientifically grounded and directly transferable to ROS/Nav2 integration later.

---

## ✔️ Scenario Overview

The system is evaluated under four sequential human messages and simulated robot reliability conditions:

1️⃣ Neutral corridor  
2️⃣ Crowded environment with children  
3️⃣ Hidden/unknown hazard ahead  
4️⃣ Slippery and dangerous corridor (Greek language input)

For each step, the framework computes:
- **Self-Healing decision**
- **Language-derived risk & uncertainty**
- **Ethical risk amplification**
- **Trust evolution over time**
- **Planner parameter adaptation**

---

## 🧪 STEP 1 — Neutral Environment

**Message:**  
`Balanced corridor, nothing special here.`

- Self-Healing → Not triggered  
- Language Safety → Neutral (`risk_scale = 1.0`)  
- Ethical Layer → Neutral  
- Trust = **0.80** (default, high confidence)
- Planner remains close to nominal values  
  → moderate velocity, normal safety radius

**Interpretation:**  
Robot remains confident, moves normally, not overly conservative.

---

## 🧪 STEP 2 — Crowded Area with Children

**Message:**  
`There are many people and children ahead.`

- Language → Strong risk increase  
  - crowding + children detected  
  - `risk_scale ≈ 2.4`, `uncertainty ↑`
- Ethical Layer → amplifies risk further  
  - `ethical_risk_scale ≈ 1.56`
- Trust = **0.85 (↑ increases!)**
  → The robot “listened” to the human warning and behaved safely.
- Planner reacts:
  - **velocity drops significantly**
  - **obstacle inflation radius reaches maximum**
  - larger spatial safety margins

**Interpretation:**  
The robot behaves **socially and ethically responsible**, prioritizing vulnerable humans and crowd safety.

---

## 🧪 STEP 3 — Hidden Hazard + Localization Drift

**Message:**  
`Be careful, hidden danger around the corner.`

- Self-Healing → **Triggered**
  - localization drift exceeded threshold
  - recommends estimator adjustment and risk increase
- Language → unseen hazard → increases uncertainty
- Ethical Layer → neutral (no children/elderly factor)
- Trust = **0.70 (↓ decreases)**  
  → system shows reliability degradation
- Planner becomes more conservative:
  - slower motion
  - larger safety buffers

**Interpretation:**  
Robot recognizes **internal weakness + external uncertainty**, becomes cautious, and explicitly explains why.

---

## 🧪 STEP 4 — Slippery & Dangerous Corridor (Greek Input)

**Message (Greek):**  
`Ο διάδρομος είναι γλιστερός και επικίνδυνος.`  
("The corridor is slippery and dangerous.")

- Language Safety:
  - high risk
  - very high uncertainty
- Self-Healing:
  - high regret + failure rate
  - **Safe Mode activated**
- Trust = **0.40 (significant drop)**  
  → appropriate distrust due to failures
- Planner:
  - **very low speed**
  - **maximum safety radius**
  - cautious behavior

**Interpretation:**  
Robot enters **maximum safety mode** because:
- environment is dangerous
- autonomy reliability is compromised
This is the desired and ethical behavior.

---

## 🎯 Key Takeaways

✔ Robot proactively adapts based on **human language input**  
✔ Robot **self-diagnoses reliability degradation** and reacts  
✔ Robot becomes **more ethical** when human vulnerability exists  
✔ Trust dynamically regulates conservativeness  
✔ Planner parameters change in realistic and meaningful ways

---

## 🔍 What These Results Mean Scientifically

These results demonstrate:
- human-centered navigation
- uncertainty-aware autonomy
- ethical risk reasoning
- trust-aware safety control
- explainable decision making

They form a solid foundation for:
- ROS2/Nav2 integration
- simulation in Gazebo
- real-robot experiments
- HRI evaluation studies
- research publications

---
