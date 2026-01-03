import csv
import argparse
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--attack_start", type=float, required=True)
    ap.add_argument("--rate_hz", type=float, default=20.0)
    args = ap.parse_args()

    t, alarm = [], []
    with open(args.csv, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t.append(float(row["t_sec"]))
            except:
                continue
            try:
                alarm.append(float(row["tf_alarm"]))
            except:
                alarm.append(np.nan)

    t = np.array(t, dtype=float)
    alarm = np.array(alarm, dtype=float)

    finite = np.isfinite(t) & np.isfinite(alarm)
    t = t[finite]
    alarm = alarm[finite]

    if t.size == 0:
        raise RuntimeError("No valid rows found.")

    # sanity: alarm before attack?
    pre = alarm[t < args.attack_start]
    pre_rate = float(np.mean(pre > 0.5)) if pre.size else 0.0

    idx = np.where((t >= args.attack_start) & (alarm > 0.5))[0]
    if idx.size == 0:
        print("No detection after attack_start.")
        print(f"pre_attack_alarm_rate={pre_rate:.3f}")
        return

    first = int(idx[0])
    # delay in steps relative to first sample at/after attack_start
    first_after_attack = int(np.where(t >= args.attack_start)[0][0])
    delay_steps = first - first_after_attack
    delay_sec = delay_steps / args.rate_hz

    print(f"first_alarm_index={first}")
    print(f"first_sample_after_attack_index={first_after_attack}")
    print(f"delay_steps={delay_steps}")
    print(f"delay_sec_est={delay_sec:.6f}")
    print(f"pre_attack_alarm_rate={pre_rate:.3f}")

if __name__ == "__main__":
    main()
