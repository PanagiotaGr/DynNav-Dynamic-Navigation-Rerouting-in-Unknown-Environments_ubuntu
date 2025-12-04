import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ukf_fusion import UKF  # κλάση UKF που ήδη έχεις


def load_vo_trajectory(csv_path: str = "vo_trajectory.csv"):
    """
    Φορτώνει την VO τροχιά από CSV.
    Υποθέτουμε ότι το αρχείο έχει τουλάχιστον στήλες: 'x', 'z'.

    - x: οριζόντια συντεταγμένη (VO frame)
    - z: χρησιμοποιείται ως 'κάθετη' συντεταγμένη στο 2D επίπεδο

    Επιστρέφει:
        xs, ys : numpy arrays με τις 2D συντεταγμένες της VO τροχιάς
    """
    df = pd.read_csv(csv_path)
    xs = df["x"].to_numpy()
    zs = df["z"].to_numpy()   # z ως 'y' στο 2D επίπεδο
    return xs, zs


def compute_yaw_from_xy(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Υπολογίζει yaw γωνία από διαδοχικές διαφορές στο επίπεδο (x, y).
    Χρησιμοποιούμε atan2(dy, dx) για να πάρουμε προσανατολισμό.

    Επιστρέφει:
        yaws: numpy array με yaw (rad) για κάθε σημείο.
    """
    dx = np.diff(xs, prepend=xs[0])
    dy = np.diff(ys, prepend=ys[0])
    yaws = np.arctan2(dy, dx)
    return yaws


def moving_average(x: np.ndarray, window: int = 31) -> np.ndarray:
    """
    Απλό smoothing με moving average.
    - Το window πρέπει να είναι περιττός ακέραιος.
    - mode='same' για να διατηρήσουμε ίδιο μήκος με το input.

    Επιστρέφει:
        smoothed: numpy array με φιλτραρισμένες τιμές.
    """
    if window < 3:
        return x.copy()
    if window % 2 == 0:
        window += 1  # κάν' το περιττό
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Υπολογισμός Root Mean Square Error μεταξύ δύο διαδρομών.
    """
    return np.sqrt(np.mean((a - b) ** 2))


if __name__ == "__main__":
    # ==========================================================
    # 0. Βασικές παράμετροι
    # ==========================================================
    dt = 0.1   # υποθέτουμε ~10Hz. Άλλαξέ το αν ξέρεις το πραγματικό rate.
    vo_csv_path = "vo_trajectory.csv"

    # ==========================================================
    # 1. Φόρτωση VO τροχιάς (x,z → (x,y) στο επίπεδο)
    # ==========================================================
    xs_vo, ys_vo = load_vo_trajectory(vo_csv_path)
    N = len(xs_vo)
    t = np.arange(N) * dt

    # ==========================================================
    # 2. Smoothed VO ως reference (pseudo-ground truth)
    # ==========================================================
    xs_ref = moving_average(xs_vo, window=31)
    ys_ref = moving_average(ys_vo, window=31)

    # ==========================================================
    # 3. Υπολογισμός yaw από την VO τροχιά
    # ==========================================================
    yaws_vo = compute_yaw_from_xy(xs_vo, ys_vo)

    # ==========================================================
    # 4. Δημιουργία pseudo-odometry (vx, vy, yaw_rate) από VO + θόρυβο
    # ==========================================================
    vx = np.diff(xs_vo, prepend=xs_vo[0]) / dt
    vy = np.diff(ys_vo, prepend=ys_vo[0]) / dt
    yaw_rate = np.diff(yaws_vo, prepend=yaws_vo[0]) / dt

    rng = np.random.default_rng(seed=42)

    # Προσθέτουμε μικρό θόρυβο για να μοιάζουν με wheel odometry
    vx_odom = vx + rng.normal(0, 0.02, size=N)
    vy_odom = vy + rng.normal(0, 0.02, size=N)
    yaw_rate_odom = yaw_rate + rng.normal(0, 0.01, size=N)

    # ==========================================================
    # 5. "Μετρήσεις" VO (pose) με επιπλέον θόρυβο
    # ==========================================================
    vo_x_meas = xs_vo + rng.normal(0, 0.05, size=N)
    vo_y_meas = ys_vo + rng.normal(0, 0.05, size=N)
    vo_yaw_meas = yaws_vo + rng.normal(0, 0.02, size=N)

    # ==========================================================
    # 6. "Μετρήσεις" IMU yaw με μικρότερο θόρυβο
    # ==========================================================
    imu_yaw = yaws_vo + rng.normal(0, 0.01, size=N)

    # ==========================================================
    # 7. Ολοκλήρωση odom χωρίς VO/IMU (για σύγκριση)
    # ==========================================================
    x_odom = [xs_vo[0]]
    y_odom = [ys_vo[0]]
    yaw_odom = [yaws_vo[0]]

    for k in range(1, N):
        x_odom.append(x_odom[-1] + vx_odom[k] * dt)
        y_odom.append(y_odom[-1] + vy_odom[k] * dt)
        yaw_odom.append(yaw_odom[-1] + yaw_rate_odom[k] * dt)

    x_odom = np.array(x_odom)
    y_odom = np.array(y_odom)

    # ==========================================================
    # 8. UKF fusion (VO + Odom + IMU)
    # ==========================================================
    ukf = UKF()
    x_est, y_est = [], []

    for k in range(N):
        # control input u = [vx, vy, yaw_rate] από "odom"
        u = [vx_odom[k], vy_odom[k], yaw_rate_odom[k]]
        ukf.predict(u=u, dt=dt)

        # VO μέτρηση (pose)
        z_vo = np.array([vo_x_meas[k], vo_y_meas[k], vo_yaw_meas[k]])
        ukf.update_vo(z_vo)

        # IMU yaw μέτρηση
        ukf.update_imu(imu_yaw[k])

        # Αποθήκευση fused εκτίμησης
        x_est.append(ukf.x[0])
        y_est.append(ukf.x[1])

    x_est = np.array(x_est)
    y_est = np.array(y_est)

    # ==========================================================
    # 9. Αποθήκευση fused τροχιάς σε CSV για downstream modules
    # ==========================================================
    fused_df = pd.DataFrame({
        "t": t,
        "x_vo": xs_vo,
        "y_vo": ys_vo,
        "x_odom": x_odom,
        "y_odom": y_odom,
        "x_ukf": x_est,
        "y_ukf": y_est,
    })
    fused_df.to_csv("ukf_fused_trajectory.csv", index=False)
    print("[INFO] Saved fused trajectory to ukf_fused_trajectory.csv")

    # ==========================================================
    # 10. Υπολογισμός RMSE ως προς το smoothed VO reference
    # ==========================================================
    rmse_vo_vs_ref = rmse(xs_vo + 0j, xs_ref + 0j) + rmse(ys_vo + 0j, ys_ref + 0j)
    rmse_odom_vs_ref = rmse(x_odom + 0j, xs_ref + 0j) + rmse(y_odom + 0j, ys_ref + 0j)
    rmse_ukf_vs_ref = rmse(x_est + 0j, xs_ref + 0j) + rmse(y_est + 0j, ys_ref + 0j)

    print(f"[STATS] RMSE VO vs smoothed ref (x+y):   {rmse_vo_vs_ref:.4f}")
    print(f"[STATS] RMSE Odom vs smoothed ref (x+y): {rmse_odom_vs_ref:.4f}")
    print(f"[STATS] RMSE UKF vs smoothed ref (x+y):  {rmse_ukf_vs_ref:.4f}")

    # ==========================================================
    # 11. Plot σύγκρισης
    # ==========================================================
    plt.figure(figsize=(7, 6))

    # Smoothed VO reference (σαν ground truth)
    plt.plot(xs_ref, ys_ref,
             label="Smoothed VO reference",
             linewidth=2)

    # Ολοκλήρωση μόνο με odom
    plt.plot(x_odom, y_odom, "--",
             label="Odom-only integration",
             linewidth=1)

    # Noisy VO measurements
    plt.scatter(vo_x_meas, vo_y_meas,
                s=6,
                alpha=0.5,
                label="Noisy VO measurements")

    # UKF fused estimate
    plt.plot(x_est, y_est,
             label="UKF fused estimate",
             linewidth=2)

    plt.axis("equal")
    plt.xlabel("X (VO units)")
    plt.ylabel("Y (VO units)")
    plt.title("UKF Fusion on Real VO Trajectory (x-z projection)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("ukf_fusion_vo_result.png", dpi=200)
    print("[INFO] Saved plot to ukf_fusion_vo_result.png")

    plt.show()

    print("UKF fusion on real VO completed.")

