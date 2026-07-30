<div align="center">

# DynNav

## Δυναμικός επανασχεδιασμός με διατήρηση ασφαλών επιλογών διαφυγής

**Ένα ερευνητικό έργο για risk- και recoverability-aware πλοήγηση σε δυναμικά και μερικώς παρατηρήσιμα περιβάλλοντα.**

[![CI](https://github.com/panagiotagrosdouli/DynNav/actions/workflows/ci.yml/badge.svg)](https://github.com/panagiotagrosdouli/DynNav/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-Apache--2.0-4C1.svg)](LICENSE)

[English](README.md) · **Ελληνικά** · [Τεχνική τεκμηρίωση](docs/README.md) · [Ερευνητικό roadmap](docs/DYNNAV_V2_RESEARCH_ROADMAP.md) · [Λοιπές επεκτάσεις](docs/CONTRIBUTION_FEATURE_CATALOG.md)

</div>

---

## Κεντρική ιδέα

Η συντομότερη διαθέσιμη διαδρομή δεν είναι πάντα η ασφαλέστερη απόφαση. Ένα ρομπότ μπορεί να εισέλθει σε στενό πέρασμα ή περιοχή με μία μόνο έξοδο και να παγιδευτεί όταν εμφανιστεί νέο εμπόδιο, αλλάξει η εκτίμηση του χάρτη ή ακυρωθεί η αρχική διαδρομή.

Το DynNav εξετάζει ένα συγκεκριμένο ερευνητικό ερώτημα:

> **Μπορεί η ρητή εκτίμηση της ανακτησιμότητας να μειώσει τις μη αναστρέψιμες αποτυχίες κατά τον online επανασχεδιασμό, χωρίς υπερβολική αύξηση του μήκους διαδρομής και του υπολογιστικού κόστους;**

Η βασική αρχή είναι:

> Ο planner δεν πρέπει να αξιολογεί μόνο αν μπορεί να φτάσει στον στόχο, αλλά και αν κάθε απόφαση διατηρεί αρκετές ασφαλείς επιλογές αντίδρασης όταν το περιβάλλον αλλάξει.

---

## Βασική συνεισφορά

Το κύριο έργο ενοποιεί τρεις λειτουργίες:

1. risk-aware route selection,
2. returnability και recoverability estimation,
3. online dynamic replanning.

Για μια διαδρομή \(\pi\), το βασικό objective είναι:

```math
J(\pi)=L(\pi)+\lambda_r R(\pi)+\lambda_{irr}I(\pi),
```

όπου:

- \(L(\pi)\): γεωμετρικό ή traversal κόστος,
- \(R(\pi)\): σωρευτική έκθεση σε occupancy ή collision risk,
- \(I(\pi)\): κόστος μη αναστρεψιμότητας.

Μια αρχική κατάσταση μπορεί να αξιολογείται ως:

```math
I(s)=w_1\frac{1}{1+N_{escape}(s)}+w_2B(s)+w_3P_{return\ failure}(s),
```

με:

- \(N_{escape}(s)\): αριθμό διαθέσιμων ή ανεξάρτητων επιλογών διαφυγής,
- \(B(s)\): έκθεση σε bottleneck,
- \(P_{return\ failure}(s)\): εκτιμώμενη πιθανότητα αποτυχίας επιστροφής σε ασφαλή περιοχή.

Οι όροι, τα εύρη και η κανονικοποίησή τους πρέπει να ορίζονται ρητά και να ελέγχονται με zero-weight ablations και sensitivity analysis.

---

## Πειραματικές υποθέσεις

- **H1:** Η recoverability-aware σχεδίαση μειώνει τις μη αναστρέψιμες αποτυχίες σε σχέση με shortest-path και risk-only planners.
- **H2:** Το όφελος αυξάνεται όσο αυξάνεται η πιθανότητα δυναμικής ακύρωσης της διαδρομής.
- **H3:** Η μείωση αποτυχιών επιτυγχάνεται με περιορισμένο path-length και runtime overhead.
- **H4:** Ο συνδυασμός risk και recoverability υπερέχει των δύο όρων όταν χρησιμοποιούνται ανεξάρτητα.

---

## Baselines και ablations

| Planner | Objective |
|---|---|
| Shortest | \(J_0=L\) |
| Risk-aware | \(J_1=L+\lambda_rR\) |
| Recoverability-aware | \(J_2=L+\lambda_{irr}I\) |
| Combined | \(J_3=L+\lambda_rR+\lambda_{irr}I\) |

Οι συγκρίσεις πρέπει να εκτελούνται στους ίδιους χάρτες, seeds, αρχικές συνθήκες και ακολουθίες δυναμικών αλλαγών.

---

## Σενάρια αξιολόγησης

Το ελάχιστο benchmark περιλαμβάνει:

- αδιέξοδα και περιοχές με μία έξοδο,
- στενά περάσματα και bottlenecks,
- δύο εναλλακτικούς διαδρόμους,
- ξαφνικό κλείσιμο της αρχικής διαδρομής,
- εμπόδιο που αποκόπτει την επιστροφή,
- σταδιακή αποκάλυψη μερικώς γνωστού χάρτη.

Το κρίσιμο σενάριο συγκρίνει μια σύντομη αλλά εύθραυστη διαδρομή με μια μακρύτερη διαδρομή που διατηρεί περισσότερες επιλογές διαφυγής.

---

## Κύριες μετρικές

- mission success rate,
- irreversible failure rate,
- recovery success rate,
- emergency-stop rate,
- minimum escape-option count,
- replanning count,
- cumulative risk exposure,
- path-length overhead,
- planning time.

Η κύρια outcome metric είναι:

```math
\text{Irreversible Failure Rate}=
\frac{\text{runs χωρίς εφικτή ασφαλή έξοδο}}
{\text{σύνολο runs}}.
```

Το βασικό trade-off είναι η μείωση των μη αναστρέψιμων αποτυχιών έναντι του πρόσθετου μήκους διαδρομής και υπολογιστικού κόστους.

---

## Τι υπάρχει ήδη

Το repository περιλαμβάνει υπάρχοντα δομικά στοιχεία που θα αξιοποιηθούν και θα ενοποιηθούν:

- deterministic A* και Dijkstra baselines,
- risk-aware grid planning,
- returnability και recoverability metrics,
- D* Lite και online replanning,
- synthetic scenario generation,
- tests, benchmark scripts, manifests και exports,
- διαδραστικό dashboard για επιθεώρηση σεναρίων και αποτελεσμάτων.

Η ερευνητική αξία θα προκύψει από την αυστηρή ενοποίηση, τις σαφείς μαθηματικές συμβάσεις, τις δίκαιες συγκρίσεις, τα multi-seed experiments και τη στατιστική ανάλυση — όχι από τον αριθμό των διαθέσιμων modules.

---

## Υπόλοιπες επεκτάσεις

Το repository διατηρεί πρόσθετες κατευθύνσεις, όπως learning, multi-robot coordination, semantic navigation, security, formal shields και neural scene representations. Αυτές δεν αποτελούν την κύρια συνεισφορά του παρόντος έργου.

Ο πλήρης κατάλογος βρίσκεται στο [`docs/CONTRIBUTION_FEATURE_CATALOG.md`](docs/CONTRIBUTION_FEATURE_CATALOG.md).

---

## Εκτέλεση

```bash
git clone https://github.com/panagiotagrosdouli/DynNav.git
cd DynNav
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"
pytest
```

Dashboard:

```bash
streamlit run app/dashboard.py
```

---

## Evidence boundaries

Το DynNav είναι ενεργό ερευνητικό έργο με πραγματικό κώδικα, tests και εκτελέσιμα πειράματα. Ωστόσο, κάθε ισχυρισμός θα συνδέεται με συγκεκριμένη έκδοση κώδικα, configuration, seeds, παραγόμενα artifacts και στατιστική ανάλυση.

Μέχρι να υπάρχουν τα αντίστοιχα αποτελέσματα, δεν δηλώνονται:

- επικύρωση σε φυσικό ρομπότ,
- production-ready ROS 2/Nav2 integration,
- τυπική εγγύηση ασφάλειας,
- γενίκευση πέρα από τα αξιολογημένα σενάρια,
- υπεροχή έναντι εξωτερικών baselines χωρίς δίκαιη σύγκριση.

Αυτά είναι όρια της διαθέσιμης απόδειξης και όχι όρια του τελικού ερευνητικού στόχου.

---

## Ερευνητική ταυτότητα

**Planning to Preserve Escape Options**

**Recoverability-Aware Dynamic Navigation in Partially Observed Environments**
