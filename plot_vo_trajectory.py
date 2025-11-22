import csv
import matplotlib.pyplot as plt

def load_trajectory(csv_path="vo_trajectory.csv"):
    frames = []
    xs, ys, zs = [], [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame_idx"]))
            xs.append(float(row["x"]))
            ys.append(float(row["y"]))
            zs.append(float(row["z"]))

    return frames, xs, ys, zs


def main():
    frames, xs, ys, zs = load_trajectory("vo_trajectory.csv")

    print(f"Loaded {len(frames)} poses")

    # 2D προβολή στο επίπεδο x-z (οριζόντια κίνηση)
    plt.figure()
    plt.plot(xs, zs, marker="o", linewidth=1)
    plt.xlabel("x (arbitrary units)")
    plt.ylabel("z (arbitrary units)")
    plt.title("Visual Odometry Trajectory (x-z projection)")
    plt.grid(True)
    plt.axis("equal")

    plt.show()


if __name__ == "__main__":
    main()
