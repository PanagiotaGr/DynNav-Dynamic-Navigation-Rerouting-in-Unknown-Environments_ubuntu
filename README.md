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

All experimental scripts and result tables are available under `experiments/`, `phd_experiments/`, and `results/statistical_runs/`.

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
