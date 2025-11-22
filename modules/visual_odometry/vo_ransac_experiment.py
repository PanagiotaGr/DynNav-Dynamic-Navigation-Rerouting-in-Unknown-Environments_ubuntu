import sys
import cv2
import numpy as np


def detect_and_match_features(img1_gray, img2_gray, nfeatures=2000):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None:
        return None, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) == 0:
        return None, None, None

    matches = sorted(matches, key=lambda m: m.distance)
    n_good = max(50, int(0.25 * len(matches)))
    good_matches = matches[:n_good]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    return pts1, pts2, good_matches


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 vo_ransac_experiment.py <video_path> <threshold>")
        sys.exit(1)

    video_path = sys.argv[1]
    threshold = float(sys.argv[2])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        sys.exit(1)

    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame")
        sys.exit(1)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    h, w = prev_gray.shape
    focal = 0.8 * w
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[focal, 0, cx],
                  [0, focal, cy],
                  [0, 0, 1]], dtype=np.float64)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pts1, pts2, matches = detect_and_match_features(prev_gray, frame_gray)

        if pts1 is None or len(pts1) < 8:
            print(f"[Frame {frame_idx}] Not enough matches")
            prev_gray = frame_gray
            frame_idx += 1
            continue

        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=threshold
        )

        if E is None or mask is None:
            print(f"[Frame {frame_idx}] Essential matrix failed")
            prev_gray = frame_gray
            frame_idx += 1
            continue

        inliers = int(mask.sum())
        print(
            f"[Frame {frame_idx}] threshold={threshold:.2f}, "
            f"inliers={inliers}/{len(pts1)}"
        )

        prev_gray = frame_gray
        frame_idx += 1

    cap.release()
    print("Done.")


if __name__ == "__main__":
    main()
