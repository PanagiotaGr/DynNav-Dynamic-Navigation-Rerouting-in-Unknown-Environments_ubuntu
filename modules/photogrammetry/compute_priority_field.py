import csv
import numpy as np
import matplotlib.pyplot as plt


def load_coverage_grid(csv_path="coverage_grid.csv"):
    rows = []
    cols = []
    center_xs = []
    center_zs = []
    covered_vals = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(int(row["row"]))
            cols.append(int(row["col"]))
            center_xs.append(float(row["center_x"]))
            center_zs.append(float(row["center_z"]))
            covered_vals.append(bool(int(row["covered"])))

    nz = max(rows) + 1
    nx = max(cols) + 1

    grid_x = np.zeros((nz, nx))
    grid_z = np.zeros((nz, nx))
    covered = np.zeros((nz, nx), dtype=bool)

    for r, c, cx, cz, cov in zip(rows, cols, center_xs, center_zs, covered_vals):
        grid_x[r, c] = cx
        grid_z[r, c] = cz
        covered[r, c] = cov

    return grid_x, grid_z, covered


def load_uncertainty_map(csv_path="feature_uncertainty_grid.csv"):
    rows = []
    cols = []
    uncertainty = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(int(row["row"]))
            cols.append(int(row["col"]))
            uncertainty.append(float(row["uncertainty"]))

    nz = max(rows) + 1
    nx = max(cols) + 1
    U = np.zeros((nz, nx))

    for (r,c,u) in zip(rows, cols, uncertainty):
        U[r,c] = u

    return U


def save_priority(grid_x, grid_z, P, path="priority_field.csv"):
    nz, nx = grid_x.shape

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row","col","center_x","center_z","priority"])
        for r in range(nz):
            for c in range(nx):
                w.writerow([
                    r, c,
                    float(grid_x[r,c]),
                    float(grid_z[r,c]),
                    float(P[r,c])
                ])

    print(f"[OK] Saved priority field to: {path}")


def plot_priority(grid_x, grid_z, P):
    nz,nx = grid_x.shape

    xmin = grid_x[0,0]
    xmax = grid_x[0,-1]
    zmin = grid_z[0,0]
    zmax = grid_z[-1,0]

    # cell size
    cell_x = abs(grid_x[0,1]-grid_x[0,0]) if nx>1 else 1
    cell_z = abs(grid_z[1,0]-grid_z[0,0]) if nz>1 else 1
    cell = max(cell_x, cell_z)

    extent = [
        xmin-0.5*cell,
        xmax+0.5*cell,
        zmin-0.5*cell,
        zmax+0.5*cell
    ]

    plt.figure()
    plt.imshow(
        P,
        origin='lower',
        extent=extent,
        interpolation='nearest',
        aspect='equal'
    )
    plt.colorbar(label="Priority value")
    plt.title("Priority Field (Uncertainty × Uncovered)")
    plt.xlabel("x (VO units)")
    plt.ylabel("z (VO units)")
    plt.grid(True)
    plt.show()


def main():
    grid_x, grid_z, covered = load_coverage_grid()
    U = load_uncertainty_map()

    nz,nx = U.shape

    C = covered.astype(np.float32)

    P = (1.0-C)*U

    maxv = P.max()
    if maxv>0:
        P /= maxv

    save_priority(grid_x, grid_z, P)

    plot_priority(grid_x, grid_z, P)


if __name__=="__main__":
    main()
