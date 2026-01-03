import csv
import math
import numpy as np
import matplotlib.pyplot as plt
import argparse

def run_cusum(score, k):
    g = np.zeros_like(score, dtype=float)
    acc = 0.0
    for i, s in enumerate(score):
        acc = max(0.0, acc + (s - k))
        g[i] = acc
    return g

def load_csv(path):
    t, score, cusum, alarm = [], [], [], []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            def to_float(x):
                try:
                    return float(x)
                except Exception:
                    return math.nan

            t.append(to_float(row.get("t_sec", "nan")))
            score.append(to_float(row.get("tf_score", "nan")))
            cusum.append(to_float(row.get("tf_cusum", "nan")))
            alarm.append(to_float(row.get("tf_alarm", "nan")))

    return np.array(t), np.array(score), np.array(cusum), np.array(alarm)

def estimate_delay(t, alarm, attack_start):
    # first alarm after attack_start
    idx = np.where((t >= attack_start) & (alarm > 0.5))[0]
    if idx.size == 0:
        return np.nan
    return float(t[idx[0]] - attack_start)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_prefix", default="tf_calib")
    ap.add_argument("--fpr_target", type=float, default=0.01)
    ap.add_argument("--k", type=float, default=None)
    ap.add_argument("--attack_start", type=float, default=None)
    args = ap.parse_args()

    t, score, _, alarm = load_csv(args.csv)

    # Keep only finite pairs (t, score)
    mask = np.isfinite(t) & np.isfinite(score)
    t = t[mask]
    score = score[mask]
    alarm = alarm[mask] if alarm.size == mask.size else np.full_like(t, np.nan)

    if score.size < 50:
        raise RuntimeError(
            "Not enough valid samples in CSV.\n"
            "Likely cause: /ids/tf_score is not being published OR logger ran too briefly.\n"
            "Fix: run the TF monitor + logger for 20–60 seconds and try again."
        )

    k = float(np.median(score)) if args.k is None else float(args.k)
    g = run_cusum(score, k)

    # Threshold h chosen so that only fpr_target fraction exceeds it (no-attack calibration)
    q = 1.0 - float(args.fpr_target)
    q = min(max(q, 0.0), 1.0)
    h = float(np.quantile(g, q))

    # Plots
    plt.figure()
    plt.plot(t, score)
    plt.xlabel("t (s)")
    plt.ylabel("tf_score")
    plt.title("TF normalized score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_score.png", dpi=160)

    plt.figure()
    plt.plot(t, g)
    plt.axhline(h)
    plt.xlabel("t (s)")
    plt.ylabel("CUSUM g")
    plt.title(f"CUSUM (k={k:.4f}) threshold h@FPR≈{args.fpr_target}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{args.out_prefix}_cusum.png", dpi=160)

    delay = np.nan
    if args.attack_start is not None:
        # alarm may contain NaNs if topic not published
        finite_alarm = np.isfinite(alarm)
        if np.any(finite_alarm):
            delay = estimate_delay(t[finite_alarm], alarm[finite_alarm], args.attack_start)

    with open(f"{args.out_prefix}_params.txt", "w") as f:
        f.write(f"cusum_k: {k}\n")
        f.write(f"cusum_h: {h}\n")
        f.write(f"fpr_target: {args.fpr_target}\n")
        if args.attack_start is not None:
            f.write(f"attack_start_sec: {args.attack_start}\n")
            f.write(f"detection_delay_sec: {delay}\n")

    print("Saved:")
    print(f"  {args.out_prefix}_score.png")
    print(f"  {args.out_prefix}_cusum.png")
    print(f"  {args.out_prefix}_params.txt")
    if args.attack_start is not None:
        print(f"Detection delay (sec): {delay}")

if __name__ == "__main__":
    main()
