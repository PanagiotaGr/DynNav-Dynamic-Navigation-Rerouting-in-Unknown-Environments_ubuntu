# DynNav

**Online επανασχεδιασμός διαδρομής με επίγνωση κινδύνου και δυνατότητας
ανάκαμψης για αυτόνομα ρομπότ σε δυναμικά, μερικώς παρατηρήσιμα περιβάλλοντα.**

[English](README.md) · [Ελληνικά](README_GR.md)

[![CI](https://github.com/panagiotagrosdouli/DynNav/actions/workflows/ci.yml/badge.svg)](https://github.com/panagiotagrosdouli/DynNav/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](pyproject.toml)
[![ROS 2](https://img.shields.io/badge/ROS_2-Jazzy-22314E)](ros2_ws/src/dynnav_nav2_cpp/README.md)
[![Άδεια](https://img.shields.io/badge/license-Apache--2.0-4C1.svg)](LICENSE)

Το DynNav είναι πειραματική πλατφόρμα ρομποτικής για τη μελέτη ενός
συγκεκριμένου ερωτήματος: μπορεί ένα σύστημα πλοήγησης να αντιδρά σε ακύρωση
της διαδρομής χωρίς να έχει δεσμεύσει το ρομπότ σε κατάσταση από την οποία η
ανάκαμψη δεν είναι πλέον εφικτή;

Το repository συνδυάζει αλγοριθμικά πρωτότυπα, ελεγχόμενα πειράματα, C++
plugin για Nav2, Gazebo benchmark harnesses και ερευνητικά interfaces που
δίνουν προτεραιότητα στα τεκμήρια. Κάθε σημαντικός ισχυρισμός πρέπει να μπορεί
να συνδεθεί με κώδικα, configuration, tests και διατηρημένα artifacts.

> **Κατάσταση:** ερευνητικό πρωτότυπο. Το DynNav δεν είναι πιστοποιημένο σύστημα
> ασφάλειας και η σημερινή ποσότητα recoverability είναι δομικό heuristic, όχι
> βαθμονομημένη πιθανότητα επιτυχούς ανάκαμψης.

## Ερευνητική συνεισφορά

Ο συμβατικός global planning βελτιστοποιεί συνήθως το γεωμετρικό μήκος ή το
κόστος του costmap. Το DynNav προσθέτει ένα δεύτερο κριτήριο: αν μια υποψήφια
απόφαση διατηρεί χρήσιμες επιλογές διαφυγής και επιστροφής μετά από αλλαγή του
περιβάλλοντος.

Για μετάβαση στο κελί `x`, η ελεγχόμενη οικογένεια planners χρησιμοποιεί

```text
cost(x) = 1 + λr · normalized_risk(x) + λi · irreversibility(x)
```

Η ίδια υλοποίηση εκθέτει τέσσερα ablations ώστε οι επιδράσεις του κινδύνου και
της recoverability να αξιολογούνται ανεξάρτητα.

| Objective | Βάρος κινδύνου | Βάρος irreversibility | Ρόλος |
|---|---:|---:|---|
| J0 — shortest | 0 | 0 | Γεωμετρικό baseline |
| J1 — risk-aware | > 0 | 0 | Ablation κινδύνου costmap |
| J2 — recoverability-aware | 0 | > 0 | Ablation δομικής δυνατότητας ανάκαμψης |
| J3 — joint | > 0 | > 0 | Προτεινόμενο κοινό objective |

Η κεντρική υπόθεση δεν παρουσιάζεται ως αποτέλεσμα: απαιτείται ακόμη δυναμική,
paired μελέτη επαρκούς στατιστικής ισχύος για να εξεταστεί αν τα J2 ή J3
μειώνουν τις recovery-infeasible failures χωρίς μη αποδεκτό πρόσθετο μήκος ή
υπολογιστικό κόστος.

## Τι έχει υλοποιηθεί

| Επίπεδο | Σημερινή υλοποίηση |
|---|---|
| Planning core | Ντετερμινιστικά A*, Dijkstra, J0–J3 recoverability-aware A* και πειράματα D* Lite |
| Μοντέλο περιβάλλοντος | Occupancy, κανονικοποιημένος κίνδυνος, αβεβαιότητα, δυναμικές ενημερώσεις εμποδίων και safe regions |
| Αξιολόγηση | Metrics για διαδρομή, κίνδυνο, irreversibility, failures, overhead, paired effects και confidence intervals |
| ROS integration | C++17 `nav2_core::GlobalPlanner` plugin για ROS 2 Jazzy/Nav2 |
| Simulation | Static planner-server και dynamic Gazebo route-invalidation protocols |
| Ερευνητικά εργαλεία | Reproducible runners, manifests, Markdown reports, FastAPI/Next.js Researcher και Streamlit lab |
| Επεκτάσεις | Καταχωρισμένα C01–C26 prototypes για learning, mapping, multi-robot, security και human/AI interaction |

Η κανονική ροή εκτέλεσης είναι:

1. παρατήρηση της κατάστασης occupancy και costmap,
2. εκτίμηση κινδύνου και τοπικής δομής ανάκαμψης,
3. υπολογισμός διαδρομής J0–J3,
4. εκτέλεση μέσω Nav2 ή του Python reference environment,
5. εφαρμογή ή αντίληψη γεγονότος που αλλάζει τη διαδρομή,
6. επανασχεδιασμός και διατήρηση του πλήρους evidence bundle.

## Σύνοψη τεκμηρίων

| Επίπεδο τεκμηρίων | Διατηρημένο αποτέλεσμα | Ερμηνεία |
|---|---|---|
| Software contracts | Tests σε Python 3.10–3.12, Ruff και strict typing του mapping core | Ελέγχει τα συμβόλαια υλοποίησης, όχι την αποτελεσματικότητα πλοήγησης |
| Ελεγχόμενες Python μελέτες | Paired, multi-seed J0–J3 protocol και raw artifacts | Κατάλληλα για algorithm debugging και διαμόρφωση υποθέσεων |
| Nav2 planner-server | 36/36 επιτυχημένα static requests με έξι planners | Τεκμηριώνει παραγωγή διαδρομής σε δύο συγκεκριμένα queries |
| Dynamic Gazebo commissioning | 8 έγκυρα trials, 7 επιτυχείς εκτελέσεις | Επιβεβαιώνει την πειραματική ροή· το δείγμα δεν επαρκεί για treatment claims |
| Φυσικό ρομπότ | Μόνο launch και safety checklist | Δεν υποστηρίζεται ακόμη ισχυρισμός hardware execution |

Για την ακριβή κατάσταση κάθε ισχυρισμού, χρησιμοποίησε τον
[πίνακα ισχυρισμών–τεκμηρίων](CLAIM_EVIDENCE_MATRIX.md). Ο
[ερευνητικός φάκελος](docs/PHD_APPLICATION_READINESS.md) δίνει σύντομη διαδρομή
αξιολόγησης για εργαστήρια, επιτροπές και ομάδες research engineering.

## Αναπαραγωγή του software evidence

Απαιτείται Python 3.10 ή νεότερη. Το ROS 2 Jazzy χρειάζεται μόνο για τις ροές
Nav2 και Gazebo.

```bash
git clone https://github.com/panagiotagrosdouli/DynNav.git
cd DynNav
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,researcher,dashboard]"
```

Γρήγορη διαδρομή reproducibility:

```bash
python scripts/run_all.py \
  --config configs/default.yaml \
  --smoke \
  --out-dir results/ci_smoke

python scripts/run_benchmarks.py \
  --config configs/default.yaml \
  --smoke \
  --out-dir results/ci_benchmarks
```

Έλεγχος του checkout:

```bash
ruff check dynnav ros2_ws/src/dynnav_nav2_benchmark
mypy dynnav/mapping --strict --no-warn-unused-ignores --show-error-codes
python -m pytest -q
python scripts/audit_markdown.py --root . --json-out results/markdown_audit.json
```

Κάθε runner αποθηκεύει το resolved configuration, τα seeds, τα raw rows, τις
συνόψεις και τα reports στον επιλεγμένο output directory. Τα failed και partial
trials παραμένουν ορατά.

## ROS 2 Jazzy / Nav2

Build και tests του κανονικού plugin:

```bash
source /opt/ros/jazzy/setup.bash
rosdep install \
  --from-paths ros2_ws/src/dynnav_nav2_cpp \
  --ignore-src --rosdistro jazzy -r -y

colcon build \
  --base-paths ros2_ws/src/dynnav_nav2_cpp \
  --packages-select dynnav_nav2_cpp
source install/setup.bash
colcon test --packages-select dynnav_nav2_cpp
colcon test-result --verbose
```

Το plugin παίρνει snapshot του global costmap κάτω από το mutex του, ελέγχει
frames και bounds, υποστηρίζει cancellation, απορρίπτει lethal goals και
επιστρέφει stamped `nav_msgs/msg/Path`.

Οι ορισμοί των πειραμάτων και οι κανόνες ερμηνείας βρίσκονται στο
[Gazebo benchmark protocol](docs/GAZEBO_BENCHMARK_PROTOCOL.md) και στο
[dynamic execution protocol](docs/DYNAMIC_EXECUTION_PROTOCOL.md).

## Ερευνητικά interfaces

Τα interfaces είναι προαιρετικές όψεις των ίδιων research contracts· δεν
αντικαθιστούν τα raw artifacts.

### DynNav Researcher

```bash
python -m uvicorn apps.api.main:app --reload --port 8000
npm --prefix apps/web ci --no-audit --no-fund
npm --prefix apps/web run dev
```

Άνοιξε το `http://localhost:3000`. Ένα ερευνητικό αίτημα μετατρέπεται σε
επεξεργάσιμο typed protocol και απαιτεί ρητή επιβεβαίωση πριν από την εκτέλεση.

### Εργαστήριο Streamlit

```bash
streamlit run app/dashboard.py
```

Άνοιξε το `http://localhost:8501` για κατασκευή scenarios, σύγκριση planners,
επιθεώρηση mapping, έλεγχο πειραμάτων και replay αποτελεσμάτων.

## Οδηγός repository

```text
dynnav/                     κανονικό Python package
  planners/                 J0–J3 search και incremental replanning
  experiments/              scenarios και ελεγχόμενες μελέτες
  evaluation/               metrics και statistical summaries
  researcher/               typed protocols και reporting

ros2_ws/src/
  dynnav_nav2_cpp/          C++17 Nav2 global-planner plugin
  dynnav_nav2_benchmark/    static και dynamic Gazebo experiments
  dynnav_turtlebot3/        simulation και hardware bringup

apps/api/                   FastAPI research API
apps/web/                   Next.js Researcher workspace
app/                        Streamlit laboratory
contributions/              C01–C26 exploratory modules
configs/                    experiment configurations
scripts/                    runners, validators και audits
tests/                      regression και research-contract tests
results/                    retained evidence και generated artifacts
docs/                       scientific και engineering documentation
paper/                      manuscript planning material
```

## Συμβόλαιο αναπαραγωγιμότητας

Ένα δημοσιεύσιμο πείραμα πρέπει να διατηρεί:

- source commit και dirty-tree state,
- ακριβή εντολή, configuration και deterministic seeds,
- map, scenario, start/goal και event definitions,
- raw per-trial data, μαζί με failures και invalid trials,
- planner paths, costmaps, timestamps και ROS logs όπου εφαρμόζεται,
- environment και package versions,
- artifact hashes και την εντολή ανάλυσης.

Το [V2 protocol](EXPERIMENT_PROTOCOL_V2.md) είναι κανονιστικό για νέα efficacy
experiments.

## Σημερινοί περιορισμοί

- Το local escape-option score δεν έχει βαθμονομηθεί σε held-out, executed
  recovery outcomes.
- Η διατηρημένη dynamic μελέτη αποτελεί commissioning evidence και όχι powered
  efficacy comparison.
- Δεν έχει τεκμηριωθεί γενίκευση πέρα από τους συγκεκριμένους χάρτες, events,
  seeds και configurations.
- Δεν υπάρχει ακόμη αποτέλεσμα αξιοπιστίας σε φυσικό ρομπότ ή artifact formal
  verification.

Οι περιορισμοί αυτοί ορίζουν την επόμενη ερευνητική εργασία: estimator
validation, powered preregistered dynamic study, πλήρη ROS reruns με immutable
manifests και μόνο μετά σταδιακό TurtleBot3 experiment.

## Συνεισφορά και citation

- [Οδηγός συνεισφοράς](CONTRIBUTING.md)
- [Πολιτική ασφάλειας](SECURITY.md)
- [Citation metadata](CITATION.cff)
- [Publication plan](PUBLICATION_PLAN.md)

Το DynNav διατίθεται με την [Apache License 2.0](LICENSE).
