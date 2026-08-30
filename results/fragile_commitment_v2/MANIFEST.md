# Fragile Commitment V2 Synthetic Run

- Source revision: `a6e032c132c2feed5968915c6f367cf2e8f69461` plus the uncommitted V2 implementation in this workspace
- Execution date: 2026-08-18
- Evidence level: LEVEL 1 — synthetic/algorithmic only
- Design: 6 conditions × 100 seeds × J0–J3 = 2,400 rows
- Command: `python benchmarks/fragile_commitment/v2_benchmark.py --seeds 100 --output results/fragile_commitment_v2/trials.csv`
- Analysis: `PYTHONPATH=. python benchmarks/fragile_commitment/v2_analysis.py results/fragile_commitment_v2/trials.csv --summary results/fragile_commitment_v2/summary.csv --contrasts results/fragile_commitment_v2/contrasts.csv`

## SHA-256

- `trials.csv`: `8518b6ae37060288ee2298e94c271821ba0c1899f6d0829dd21322a5f1ba1392`
- `summary.csv`: `007b2af14a551a402ebd5dfacbdf98673b819fb86512c940be1db61c26a3cb09`
- `contrasts.csv`: `86b351c8d8af834a5ecf1d85a258f7790c6eb1d890d7ae17ffaf0d2055e5f255`

## Interpretation boundary

This run verifies deterministic pairing, independent event assignment, complete
J0–J3 execution, and analysis output. It is not Nav2, Gazebo, or real-robot
evidence. Several conditions produced zero J0/J2 or J1/J3 difference. Those
negative results are retained and show that the present local heuristic does
not create a treatment contrast in every intended family.
