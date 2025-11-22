import sys
import cv2
import numpy as np


def detect_and_match_features(img1_gray, img2_gray, nfeatures=1000):
    """
    Ανίχνευση χαρακτηριστικών με ORB και αντιστοίχιση με BFMatcher.
    Επιστρέφει keypoints + good matches.
    """
    orb = cv2.ORB_create(nfeatures=nfeatures)

    kp1, des1 = orb.detectAndCompute(img1_gray, None)
    kp2, des2 = orb.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None:
        return None, None, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    if len(matches) == 0:
        return None, None, None

    # ταξινόμηση κατά απόσταση (quality)
    matches = sorted(matches, key=lambda m: m.distance)

    # κρατάμε top 25% ή τουλάχιστον 50
    n_good = max(50, int(0.25 * len(matches)))
    good_matches = matches[:n_good]

    return kp1, kp2, good_matches


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 modules/visual_odometry/vo_feature_matches.py <video_path>")
        print("Example: python3 modules/visual_odometry/vo_feature_matches.py dummy_vo.avi")
        sys.exit(1)

    video_path = sys.argv[1]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        sys.exit(1)

    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame")
        sys.exit(1)

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_idx = 0

    print("Press ESC to quit, any other key to go to next frame pair.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        kp1, kp2, matches = detect_and_match_features(prev_gray, frame_gray)

        if kp1 is None or matches is None:
            print(f"[Frame {frame_idx}] Not enough matches.")
            prev_gray = frame_gray
            frame_idx += 1
            continue

        print(f"[Frame {frame_idx}] Matches: {len(matches)}")

        # οπτικοποίηση matches
        match_vis = cv2.drawMatches(
            prev_gray, kp1,
            frame_gray, kp2,
            matches, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )

        cv2.imshow("Feature matches (prev vs current)", match_vis)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break

        prev_gray = frame_gray
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

