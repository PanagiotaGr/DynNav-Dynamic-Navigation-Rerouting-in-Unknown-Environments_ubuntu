import csv
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
    print(f"Loaded {len(frames)} poses from vo_trajectory.csv")

    fig, ax = plt.subplots()

    trajectory_line, = ax.plot([], [], marker=None)
    current_point, = ax.plot([], [], marker="o")

    ax.set_xlabel("x (arbitrary units)")
    ax.set_ylabel("z (arbitrary units)")
    ax.set_title("Visual Odometry Trajectory (x-z) - Animation")
    ax.grid(True)

    if xs and zs:
        margin = 0.1
        xmin, xmax = min(xs), max(xs)
        zmin, zmax = min(zs), max(zs)
        dx = xmax - xmin if xmax != xmin else 1.0
        dz = zmax - zmin if zmax != zmin else 1.0
        ax.set_xlim(xmin - margin * dx, xmax + margin * dx)
        ax.set_ylim(zmin - margin * dz, zmax + margin * dz)

    def init():
        trajectory_line.set_data([], [])
        current_point.set_data([], [])
        return trajectory_line, current_point

    def update(i):
        xs_i = xs[: i + 1]
        zs_i = zs[: i + 1]

        trajectory_line.set_data(xs_i, zs_i)

        # IMPORTANT FIX: must be lists, not floats
        current_point.set_data([xs_i[-1]], [zs_i[-1]])

        return trajectory_line, current_point

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(xs),
        init_func=init,
        interval=50,
        blit=True,
        repeat=True,
    )

    plt.show()


if __name__ == "__main__":
    main()

