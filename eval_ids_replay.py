import argparse
import os
import numpy as np
import pandas as pd

from ukf_fusion import UKF
from attack_injector import AttackInjector, AttackConfig


# -------------------------------------------------
# Utils
# -------------------------------------------------
def ensure_dirs():
    os.makedirs("results/ids/runs", exist_ok=True)
    os.makedirs("results/ids/summary", exist_ok=True)


def safe_ratio(a, b, eps=1e-9):
    return float(a) / float(max(b, eps))


def _maybe_set_ids_params(ukf: UKF, alpha, n_consec):
    if alpha is None and n_consec is None:
        return

    candidates = []
    for name in ["ids", "monitor", "security_monitor", "ids_monitor"]:
        if hasattr(ukf, name):
            candidates.append(getattr(ukf, name))
    candidates.append(ukf)

    for obj in candidates:
        if hasattr(obj, "set_params"):
            try:
                obj.set_params(alpha=alpha, n_consecutive=n_consec)
                return
            except Exception:
                pass

        if alpha is not None:
            for a in ["alpha", "chi2_alpha"]:
                if hasattr(obj, a):
                    try:
                        setattr(obj, a, float(alpha))
                    except Exception:
                        pass

        if n_consec is not None:
            for a in ["n_consecutive", "consecutive_violations", "N"]:
                if hasattr(obj, a):
                    try:
                        setattr(obj, a, int(n_consec))
                    except Exception:
                        pass


def parse_vec3(s: str):
    p = [float(x) for x in s.split(",")]
    if len(p) != 3:
        raise argparse.ArgumentTypeError("Expected 3 values: x,y,yaw")
    return np.array(p, dtype=float)


# -------------------------------------------------
# Experiment
# -------------------------------------------------
def run_attack(
    seed=0,
    T=200,
    t_start=100,
    mode="replay",
    replay_len=30,
    bias=None,
    rate=None,
    sigma=0.0,
    duration=20,
    alpha=None,
    n_consec=None,
):
    np.random.seed(seed)
    ensure_dirs()

    # UKF
    try:
        ukf = UKF(alpha=alpha, n_consecutive=n_consec)
    except TypeError:
        ukf = UKF()
        _maybe_set_ids_params(ukf, alpha, n_consec)

    # Ground truth
    x_true = np.zeros(3)
    dt = 1.0
    u = np.array([0.05, 0.02, 0.01])

    # Attacker
    attacker = AttackInjector(AttackConfig(
        mode=mode,
        t_start=t_start,
        replay_len=replay_len,
        bias=bias,
        rate=rate,
        sigma=sigma,
        duration=duration,
    ))

    detected_at = None
    flagged_steps = 0
    triggered_steps = 0

    # -------- LOG PER TIMESTEP --------
    rows = []

    for t in range(T):
        # true motion
        x_true = x_true + u

        ukf.predict(u, dt)

        # clean VO
        z_clean = x_true + np.random.multivariate_normal(
            np.zeros(3), ukf.R_vo_nominal
        )

        attacker.observe_clean(z_clean)
        z_vo, attack_on = attacker.apply(t, z_clean)

        yaw_meas = float(
            x_true[2] + np.random.normal(0.0, np.sqrt(ukf.R_imu_nominal[0, 0]))
        )

        ukf.update_vo(z_vo)
        ukf.update_imu(yaw_meas)

        info = ukf.last_vo_ids

        if info["flagged"]:
            flagged_steps += 1
        if info["triggered"]:
            triggered_steps += 1
        if ukf.security_alert and detected_at is None:
            detected_at = t

        rows.append({
            "t": t,
            "mode": mode,
            "seed": seed,
            "attack_on": attack_on,
            "d2": info["d2"],
            "thr": info["thr"],
            "ratio": safe_ratio(info["d2"], info["thr"]),
            "flagged": info["flagged"],
            "triggered": info["triggered"],
            "streak": info["streak"],
            "scale": ukf.vo_scale_last,
            "security_alert": ukf.security_alert,
        })

        if (t % 10 == 0) or (t_start - 3 <= t <= t_start + 8):
            print(
                f"t={t:03d} mode={mode:10s} attack={'YES' if attack_on else 'no '} "
                f"d2={info['d2']:.2f} thr={info['thr']:.2f} "
                f"scale={ukf.vo_scale_last:.2f} alert={ukf.security_alert}"
            )

    # -------- SAVE PER-RUN CSV --------
    df = pd.DataFrame(rows)

    run_name = (
        f"ids_mode={mode}_alpha={alpha}_N={n_consec}_seed={seed}.csv"
    )
    df.to_csv(f"results/ids/runs/{run_name}", index=False)

    # -------- SUMMARY --------
    delay = None if detected_at is None else detected_at - t_start
    post = df[df["t"] >= t_start]

    summary = {
        "mode": mode,
        "seed": seed,
        "alpha": alpha,
        "N": n_consec,
        "t_attack": t_start,
        "t_detect": detected_at,
        "delay": delay,
        "flag_rate": flagged_steps / T,
        "trigger_rate": triggered_steps / T,
        "mean_scale_post": post["scale"].mean(),
        "max_scale_post": post["scale"].max(),
    }

    pd.DataFrame([summary]).to_csv(
        f"results/ids/summary/summary_{run_name}", index=False
    )

    return summary


# -------------------------------------------------
# CLI
# -------------------------------------------------
def main():
    p = argparse.ArgumentParser()

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--T", type=int, default=200)
    p.add_argument("--t_start", type=int, default=100)

    p.add_argument("--mode", choices=["none", "replay", "step_bias", "ramp_bias", "burst_noise"],
                   default="replay")
    p.add_argument("--replay_len", type=int, default=30)

    p.add_argument("--bias", type=parse_vec3, default=None)
    p.add_argument("--rate", type=parse_vec3, default=None)

    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--duration", type=int, default=20)

    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--n", type=int, default=None)

    args = p.parse_args()

    out = run_attack(
        seed=args.seed,
        T=args.T,
        t_start=args.t_start,
        mode=args.mode,
        replay_len=args.replay_len,
        bias=args.bias,
        rate=args.rate,
        sigma=args.sigma,
        duration=args.duration,
        alpha=args.alpha,
        n_consec=args.n,
    )

    print("\nSUMMARY:", out)


if __name__ == "__main__":
    main()
