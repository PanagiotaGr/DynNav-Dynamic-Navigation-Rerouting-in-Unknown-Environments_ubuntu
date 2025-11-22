import time
import csv
from collections import deque

# === PARAMETERS ===
INLIER_TRIGGER_THRESHOLD = 40        # if mean(last N frames) < this ⇒ replan
WINDOW = 10                          # N frames to average
REPLAN_COOLDOWN = 15                 # frames to wait after each replan
DUMMY_SPEED = 1.0                    # not used now, placeholder for future


class AdaptivePlanner:
    def __init__(self):
        self.inlier_hist = deque(maxlen=WINDOW)
        self.cooldown = 0
        self.current_route_id = 0

    def record_inliers(self, count: int):
        """Προσθέτει την τιμή inliers στο moving window."""
        self.inlier_hist.append(count)

    def should_trigger_replan(self) -> bool:
        """Επιστρέφει True αν πρέπει να γίνει replan βάσει ιστορικού inliers."""
        # Cooldown – αν μόλις κάναμε replan, περιμένουμε REPLAN_COOLDOWN frames
        if self.cooldown > 0:
            self.cooldown -= 1
            return False

        # Χρειαζόμαστε τουλάχιστον WINDOW δείγματα για στατιστική
        if len(self.inlier_hist) < WINDOW:
            return False

        mean_inliers = sum(self.inlier_hist) / len(self.inlier_hist)
        print(f"[DEBUG] mean_inliers over last {WINDOW} frames = {mean_inliers:.2f}")

        return mean_inliers < INLIER_TRIGGER_THRESHOLD

    def do_replan(self):
        print(">>> REPLAN TRIGGERED!")

        # import μόνο όταν χρειαστεί – ΠΡΟΣΟΧΗ: διορθωμένο import
        from modules.photogrammetry import replan_weighted as rw

        rw.main()  # τρέχει το weighted replan

        self.current_route_id += 1
        self.cooldown = REPLAN_COOLDOWN
        print(f"[INFO] Replan done. New route id = {self.current_route_id}, cooldown={self.cooldown}")


def simulate_live_loop():
    """
    Demo συνάρτηση: "παριστάνει" ένα live σύστημα που λαμβάνει inliers ανά frame.
    Εδώ, για απλότητα, διαβάζουμε inliers από vo_stats.csv αντί για πραγματικό live VO.
    """
    planner = AdaptivePlanner()

    inliers = []
    with open("vo_stats.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            inliers.append(int(row["num_inliers"]))

    print(f"[INFO] loaded {len(inliers)} frames from vo_stats.csv")

    for frame_idx, count in enumerate(inliers):
        print(f"[Frame {frame_idx}] inliers={count}")

        planner.record_inliers(count)

        if planner.should_trigger_replan():
            planner.do_replan()

        # μικρό delay για να μοιάζει με πραγματικό χρόνο
        time.sleep(0.05)


def main():
    simulate_live_loop()


if __name__ == "__main__":
    main()
