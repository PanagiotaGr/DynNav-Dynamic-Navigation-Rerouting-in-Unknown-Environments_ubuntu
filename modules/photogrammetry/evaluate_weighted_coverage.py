import csv
import math
import numpy as np


def load_priority_field(csv_path="priority_field.csv"):
    rows = []
    priority = []
    cx = []
    cz = []
    with open(csv_path,"r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((int(row["row"]),int(row["col"])))
            priority.append(float(row["priority"]))
            cx.append(float(row["center_x"]))
            cz.append(float(row["center_z"]))
    return rows, np.array(priority), np.array(cx), np.array(cz)


def load_weighted_path(path="replan_weighted_waypoints.csv"):
    xs = []
    zs = []
    with open(path,"r") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row["x"]))
            zs.append(float(row["z"]))
    return np.array(xs), np.array(zs)


def load_coverage(csv_path="coverage_grid.csv"):
    rows=[]
    cov=[]
    with open(csv_path,"r") as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append((int(row["row"]),int(row["col"])))
            cov.append(bool(int(row["covered"])))
    return rows, np.array(cov)


def L2(a,b):
    return math.dist(a,b)


def main():

    g_cells, covered = load_coverage()
    p_cells, P, pcx, pcz = load_priority_field()
    wx, wz = load_weighted_path()

    if len(wx)==0:
        print("[ERROR] no weighted replan found.")
        return

    # high-priority mask
    mask = P>=0.4

    hp_count = mask.sum()
    if hp_count==0:
        print("no high priority cells")
        return

    # BEFORE
    hp_cov_before = covered[mask].mean()

    # AFTER
    cov_after = covered.copy()

    # mark visited HP cells covered if close to wp
    for i,(hp_idx,val) in enumerate(zip(p_cells,P)):
        if not mask[i]:
            continue
        cx=pcx[i]
        cz=pcz[i]
        dists = np.sqrt((wx-cx)**2+(wz-cz)**2)
        if dists.min()<2.0: # visiting threshold
            cov_after[i] = True

    hp_cov_after = cov_after[mask].mean()

    gain = hp_cov_after - hp_cov_before

    # path length
    total = 0
    for i in range(1,len(wx)):
        total += L2((wx[i],wz[i]), (wx[i-1],wz[i-1]))


    print("=== Weighted Coverage Evaluation ===")
    print(f"High-priority cells: {hp_count}")
    print(f"Coverage BEFORE: {100*hp_cov_before:.2f}%")
    print(f"Coverage AFTER:  {100*hp_cov_after:.2f}%")
    print(f"Absolute improvement: {100*gain:.2f} % points")
    print(f"Weighted efficiency: {gain/total:.6f} coverage per meter")

    # save report
    with open("weighted_eval_report.csv","w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["metric","value"])
        w.writerow(["hp_cells",hp_count])
        w.writerow(["before",hp_cov_before])
        w.writerow(["after",hp_cov_after])
        w.writerow(["absolute_gain",gain])
        w.writerow(["path_length",total])
        w.writerow(["efficiency",gain/total])

    print("[OK] Saved weighted_eval_report.csv")


if __name__=="__main__":
    main()
