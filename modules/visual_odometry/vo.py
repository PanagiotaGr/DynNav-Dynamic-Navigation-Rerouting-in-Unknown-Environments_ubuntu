import sys
import cv2
import numpy as np
import csv


def detect_and_match_features(img1_gray, img2_gray):
    """
    Ανίχνευση & αντιστοίχιση χαρακτηριστικών μεταξύ δύο εικόνων
    με ORB + Brute-Force Hamming matcher.
    """
    orb = cv2.ORB_create(2000)

    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None:
        return None, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) == 0:
        return None, None, None

    # ταξινόμηση κατά απόσταση (ποιότητα)
    matches = sorted(matches, key=lambda m: m.distance)

    # κρατάμε το καλύτερο 25% ή τουλάχιστον 50 matches
    n_good = max(50, int(0.25 * len(matches)))
    good_matches = matches[:n_good]

    return kp1, kp2, good_matches


def visual_odometry(video_path: str, scale_factor: float = 1.0):
    """
    Απλή monocular visual odometry:
    - ανίχνευση χαρακτηριστικών
    - εκτίμηση Essential matrix
    - recoverPose -> R, t
    - συσσώρευση τροχιάς (up to scale)
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape

    # Απλοποιημένο intrinsics matrix
    focal = 0.8 * w
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[focal, 0, cx],
                  [0, focal, cy],
                  [0, 0, 1]], dtype=np.float64)

    # Αρχική στάση
    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros((3, 1), dtype=np.float64)

    trajectory = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp1, kp2, matches = detect_and_match_features(prev_gray, frame_gray)
        if kp1 is None or matches is None or len(matches) < 8:
            print(f"[Frame {frame_idx}] Not enough matches ({0 if matches is None else len(matches)})")
            prev_gray = frame_gray
            frame_idx += 1
            continue

        # φτιάχνουμε τα αντίστοιχα σημεία
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Υπολογισμός Essential matrix με RANSAC
        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0,
        )

        if E is None:
            print(f"[Frame {frame_idx}] Essential matrix not found")
            prev_gray = frame_gray
            frame_idx += 1
            continue

        # Ανακατασκευή σχετικής κίνησης (R, t)
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

        # Συσσώρευση (up to scale)
        t_scaled = scale_factor * t
        t_total = t_total + R_total @ t_scaled
        R_total = R @ R_total

        x, y, z = t_total.flatten()
        print(f"[Frame {frame_idx}] Estimated position: x={x:.3f}, y={y:.3f}, z={z:.3f}")

        trajectory.append((frame_idx, x, y, z))

        prev_gray = frame_gray
        frame_idx += 1

    cap.release()

    # Αποθήκευση τροχιάς σε CSV
    with open("vo_trajectory.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "x", "y", "z"])
        for row in trajectory:
            writer.writerow(row)

    print(f"[INFO] Saved trajectory to vo_trajectory.csv")
    print(f"[INFO] Total poses estimated: {len(trajectory)}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 modules/visual_odometry/vo.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    visual_odometry(video_path)


if __name__ == "__main__":
    main()
