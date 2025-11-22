import sys
import csv
import cv2
import numpy as np


def detect_and_match_features(img1_gray, img2_gray, nfeatures=2000):
    """
    Ανίχνευση ORB features και αντιστοίχιση με BFMatcher.
    Επιστρέφει (kp1, kp2, matches).
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)

    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None:
        return None, None, []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda m: m.distance)
    return kp1, kp2, matches


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 modules/visual_odometry/vo_stats.py <video_path>")
        print("Example: python3 modules/visual_odometry/vo_stats.py dummy_vo.avi")
        sys.exit(1)

    video_path = sys.argv[1]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        sys.exit(1)

    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame.")
        sys.exit(1)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape

    # απλό intrinsics μοντέλο, ίδιο στυλ με vo.py
    focal = 0.8 * w
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[focal, 0, cx],
                  [0, focal, cy],
                  [0, 0, 1]], dtype=np.float64)

    rows = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp1, kp2, matches = detect_and_match_features(prev_gray, frame_gray)

        num_matches = len(matches)
        num_inliers = 0

        if num_matches >= 8:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            # Essential + RANSAC
            E, mask = cv2.findEssentialMat(
                pts1, pts2, K,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.0
            )
            if E is not None and mask is not None:
                num_inliers = int(mask.sum())

        rows.append({
            "frame_idx": frame_idx,
            "num_matches": num_matches,
            "num_inliers": num_inliers,
        })

        print(f"[Frame {frame_idx}] matches={num_matches}, inliers={num_inliers}")

        prev_gray = frame_gray
        frame_idx += 1

    cap.release()

    # γράφουμε τα stats σε CSV
    with open("vo_stats.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_idx", "num_matches", "num_inliers"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[INFO] Saved VO stats to vo_stats.csv (frames: {len(rows)})")


if __name__ == "__main__":
    main()
