# Dynamic Navigation and Re-routing in Unknown Environments

This repository contains my individual research project for the  
**School of Electrical and Computer Engineering, Democritus University of Thrace (D.U.Th.)**.

The project focuses on **autonomous navigation, coverage planning and uncertainty-aware replanning** in unknown environments, combining:

- ROS 2 & Gazebo (TurtleBot3 simulation)
- LiDAR-based obstacle avoidance
- SLAM-based mapping and localization
- Visual Odometry (VO)
- Photogrammetry-inspired coverage planning
- Dynamic replanning based on coverage and VO uncertainty

---

## Project Structure

(main Python-side modules relevant to the analysis)

```text
modules/
  photogrammetry/
    path_planner.py              # basic lawnmower coverage path for a rectangular AOI
    path_planner_poly.py         # coverage path for polygonal AOI with no-fly zones
    coverage_map.py              # projects VO trajectory onto a grid and builds coverage_grid.csv
    replan_missing_cells.py      # replans over uncovered cells (full coverage replan)
    feature_uncertainty_map.py   # builds VO-based feature density & uncertainty maps
    compute_priority_field.py    # combines coverage & uncertainty into a priority field
    replan_weighted.py           # weighted replan using the priority field
    plot_replan_overlay.py       # overlay: coverage + VO + replan
    plot_weighted_replan_overlay.py  # overlay: coverage + VO + weighted replan
    evaluate_coverage_improvement.py # coverage metrics before/after replan
    evaluate_weighted_coverage.py    # coverage metrics for high-priority cells
    optimize_replan_cost.py      # simple multi-objective analysis over thresholds
    adaptive_replan.py           # demo of online, inlier-triggered replanning

  visual_odometry/
    vo.py                        # monocular VO pipeline (ORB + Essential matrix + trajectory export)
    vo_feature_matches.py        # (optional) visualization of feature matches
    vo_stats.py                  # per-frame inlier statistics → vo_stats.csv
