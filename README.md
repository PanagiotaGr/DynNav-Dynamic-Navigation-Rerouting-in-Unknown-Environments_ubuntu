# Dynamic Navigation and Uncertainty-Aware Replanning in Unknown Environments

This repository presents a full research-oriented pipeline for **autonomous robotic navigation in unknown environments under sensing and localization uncertainty**. The project integrates **ROS 2, Visual Odometry (VO), SLAM-based mapping, coverage planning, information gain exploration, uncertainty-aware replanning, and learned A* heuristics**, with extensive **statistical validation and ablation studies**.

The work was developed as an individual research project at the **School of Electrical and Computer Engineering, Democritus University of Thrace (D.U.Th.)**.

---

## 1. Research Problem

Autonomous navigation in unknown environments is fundamentally limited by:

* **Sensor uncertainty** (especially visual odometry drift),
* **Incomplete maps** during exploration,
* **Dynamic obstacles and replanning requirements**,
* **Trade-offs between optimality, coverage, safety, and computational cost**.

Classical global planners (A*, RRT*) assume reliable state estimation and static cost maps. However, in realistic robotic scenarios, **pose drift, feature sparsity, and tracking failures directly affect planning quality**. This project studies how navigation performance can be improved by **explicitly modeling uncertainty and learning data-driven heuristics for search-based planning**.

---

## 2. Main Contributions

The main scientific contributions of this project are:

* **Uncertainty-aware coverage planning** using VO-derived feature density and pose uncertainty.
* **Drift-weighted dynamic replanning** based on a learned priority field.
* **Information Gain (IG) and Next-Best-View (NBV) exploration** under uncertainty.
* **Learned neural heuristic for A*** to reduce node expansions and planning time.
* **Multi-objective navigation** combining entropy, coverage, path length, and safety.
* **Statistical validation** with benchmark comparisons, ablation studies, and t-tests.
* **Full ROS 2 and Gazebo integration** using TurtleBot3 simulation.

---

## 3. System Architecture

The overall pipeline is structured as:

1. **SLAM + Visual Odometry** → pose estimation and uncertainty
2. **Coverage grid projection** of robot trajectory
3. **Uncertainty and feature density mapping**
4. **Priority field construction** from coverage + uncertainty
5. **Weighted dynamic replanning**
6. **Information Gain & NBV planning**
7. **Learned A* heuristic integration**
8. **Multi-objective optimization and smoothing**
9. **Benchmark evaluation and statistical validation**

---

## 4. Key Modules

### Photogrammetry-Inspired Coverage Planning

Located in `modules/photogrammetry/`:

* Coverage path planning for rectangular and polygonal AOIs
* Missing-cell replanning
* Uncertainty-weighted priority fields
* Adaptive online replanning
* Coverage improvement evaluation

### Visual Odometry

Located in `visual_odometry/`:

* Monocular ORB-based VO
* Essential matrix pose recovery
* Inlier statistics and drift estimation

### Uncertainty Modeling

* EKF / UKF-based sensor fusion
* Pose uncertainty propagation
* Entropy and uncertainty contour modeling

### Learned A* Heuristic

* Neural regression heuristic for A*
* Curriculum training and dataset sweeps
* Heuristic uncertainty modeling
* Benchmarking vs classical A*

### Information Gain & Multi-Objective Planners

* NBV selection using entropy
* Pareto-front multi-objective navigation
* Weighted replanning under uncertainty

---

## 5. Experimental Evaluation

The repository includes a full experimental framework with:

* Multi-seed benchmark runs  
* Statistical summaries  
* Paired t-tests  
* Ablation studies  
* Confidence interval boxplots  
* Coverage, path length, replanning rate, entropy reduction, and timing metrics  

All experimental scripts and result tables are organized under:

* `experiments/`  
* `results/statistical_runs/`  

ensuring full reproducibility of the reported results.

---

## 6. Simulation Environment

* **ROS 2** middleware
* **Gazebo** simulator
* **TurtleBot3** mobile robot
* LiDAR-based obstacle avoidance
* Dynamic map updates and replanning

---

## 7. Applications

This work is applicable to:

* Autonomous mobile robots
* UAV and aerial exploration
* Search and rescue robotics
* Inspection and mapping
* Active perception systems

---

## 8. Current Status

This is an **active research project** with ongoing development in:

* Learned uncertainty modeling
* Vision-Language-Action (VLA) planning
* Online heuristic adaptation
* Drift-aware exploration policies

---

## 9. Reproducibility

To install dependencies:

```bash
pip install -r requirements.txt
```

Key entry points for experiments include:

* `train_heuristic.py`
* `eval_astar_learned.py`
* `multiobj_planner.py`
* `ig_planners_demo.py`

---

## 10. Author

**Panagiota Grosdouli**
Electrical & Computer Engineering, D.U.Th.

---

## 11. Disclaimer

This repository is intended for **research and educational use only**. Trained neural models and experimental datasets are provided for reproducibility.


## 12. Uncertainty-Aware Navigation Extensions

This repository also includes a set of research extensions that build on
top of the core dynamic navigation pipeline, with a focus on
uncertainty-aware planning, risk budgeting, and drift-aware exploration.

### 12.1 Belief–Risk Planning and λ-Sweep

We formulate global path planning as a trade-off between geometric path
length and integrated uncertainty. Given a fused uncertainty grid, we
run an A* planner with a weighted cost:

\[
J(\lambda) = L_{\text{geom}} + \lambda \, R_{\text{fused}},
\]

where \(L_{\text{geom}}\) is the geometric path length and
\(R_{\text{fused}}\) is the integrated fused risk along the path
(e.g. combining coverage and pose/VO uncertainty).

The script:

- `sweep_lambda_fused_risk.py`

performs a sweep over λ and stores metrics in:

- `belief_risk_lambda_sweep.csv`

Post-processing and visualization are handled by:

- `plot_belief_risk_lambda_sweep.py`

which generates:

- `lambda_sweep_geometric_length.png`  
- `lambda_sweep_length_cells.png`  
- `lambda_sweep_fused_risk.png`  
- `lambda_sweep_total_cost.png`  

These plots illustrate how increasing λ induces a “phase transition”
from purely shortest-path behavior to explicitly risk-averse behavior,
revealing operating regimes where risk can be reduced without changing
geometric path length.

We also support **risk-budgeted planning** via:

- `select_lambda_for_risk_budget.py`

Given a user-specified risk budget \(R_{\max}\), this script selects the
smallest λ such that the integrated fused risk satisfies
\(R(\lambda) \le R_{\max}\). This provides a simple interface for
designing planners that operate under explicit safety or risk
constraints.

---

### 12.2 Learned vs Calibrated Uncertainty on Fixed Paths

To study the effect of uncertainty calibration on path-wise risk
assessment, we compare learned uncertainty grids against their
calibrated counterparts on fixed trajectories.

The analysis script:

- `analyze_learned_vs_calib_paths.py`

produces:

- `learned_vs_calib_path_metrics.csv`

which contains integrated, mean and maximum risk values for multiple
scenarios (learned vs calibrated, evaluated crosswise on both grids).

For visualization:

- `plot_learned_vs_calib_paths.py`

generates:

- `learned_vs_calib_sum.png`  
- `learned_vs_calib_mean.png`  
- `learned_vs_calib_max.png`  

These plots highlight how calibration reshapes the risk distribution
along identical geometric paths, typically reducing overconfident or
underconfident regions while preserving the underlying structure of the
trajectories.

---

### 12.3 Drift-Aware Next-Best-View (NBV)

We extend the information-gain-based exploration framework with a
drift-aware Next-Best-View (NBV) criterion. Given:

- a coverage grid,
- an information/entropy proxy (e.g. feature density or uncertainty),
- a per-cell pose/drift uncertainty map,


Single-step comparison is implemented in:

- `drift_aware_nbv_experiment.py`

which outputs:

- `nbv_classical_topk.csv`, `nbv_classical_topk.png`  
- `nbv_driftaware_topk.csv`, `nbv_driftaware_topk.png`  

These results visualize the top-K NBV targets selected by the classical
vs drift-aware criteria, showing how drift-aware exploration shifts
attention away from highly uncertain regions while still prioritizing
informative viewpoints.

---

### 12.4 Multi-Step Drift-Aware Exploration

To evaluate exploration policies over multiple NBV decisions, we run
multi-step experiments where a policy repeatedly:

1. selects an NBV target according to its scoring rule,
2. marks the corresponding cell as covered,
3. zeroes out its local information gain.

We compare:

- **Classical policy**: \(\alpha = 0\) (pure IG-based NBV),  
- **Drift-aware policy**: \(\alpha > 0\) (information gain with drift
  penalty).

The multi-step experiment is implemented in:

- `drift_aware_nbv_multistep.py`

This script produces:

- `nbv_multistep_classical_metrics.csv` (step, IG, drift)  
- `nbv_multistep_driftaware_metrics.csv`  
- `nbv_multistep_classical_goals.csv`  
- `nbv_multistep_driftaware_goals.csv`  
- `nbv_multistep_ig.png` (information gain per step)  
- `nbv_multistep_drift.png` (drift exposure per step)  



