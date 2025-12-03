import numpy as np
import os


class HeuristicLogger:
    """
    Logger για online εκπαίδευση του heuristic.

    Αποθηκεύει:
      - features: 1D np.array (π.χ. 11D rich features)
      - targets: ground-truth remaining cost από classic A*
        (μέσω true_cost_map[(x,y)])
    """

    def __init__(self, out_path="heuristic_logs.npz", true_cost_map=None):
        self.out_path = out_path
        self.true_cost_map = true_cost_map  # dict[(x,y)] -> remaining_cost
        self.X_list = []
        self.y_list = []

    def record(self, features, node):
        """
        features: 1D np.array
        node: (x, y) grid coordinate

        Target = ground-truth remaining cost από το classic path.
        Αν το node δεν είναι στο true_cost_map, το αγνοούμε.
        """
        if self.true_cost_map is None:
            return

        if node not in self.true_cost_map:
            return

        target = float(self.true_cost_map[node])

        self.X_list.append(features.astype(np.float32))
        self.y_list.append(target)

    def save(self):
        """
        Αποθηκεύει τα μαζεμένα δείγματα σε .npz.
        Αν υπάρχει ήδη αρχείο, κάνει append.
        """
        if len(self.X_list) == 0:
            print("[LOGGER] No samples to save.")
            return

        X_new = np.stack(self.X_list, axis=0)
        y_new = np.array(self.y_list, dtype=np.float32)

        if os.path.exists(self.out_path):
            old = np.load(self.out_path)
            X_old = old["X"]
            y_old = old["y"]
            X = np.concatenate([X_old, X_new], axis=0)
            y = np.concatenate([y_old, y_new], axis=0)
        else:
            X = X_new
            y = y_new

        np.savez(self.out_path, X=X, y=y)
        print(f"[LOGGER] Saved {X.shape[0]} total samples to {self.out_path}")

        # καθαρισμός
        self.X_list = []
        self.y_list = []
