import cv2
import numpy as np

# Όνομα αρχείου βίντεο
output_file = "dummy_vo.avi"

# Διαστάσεις frame
width, height = 640, 480
fps = 20
num_frames = 200

# VideoWriter: XVID codec + AVI container
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

for i in range(num_frames):
    # Λευκό background
    frame = np.full((height, width, 3), 255, dtype=np.uint8)

    # Κινούμενο "ρομπότ" (ένα μπλε τετράγωνο που μετακινείται δεξιά)
    x = 20 + i * 2       # όσο αυξάνει το i, πάει δεξιά
    y = height // 2 - 25
    x2 = x + 50
    y2 = y + 50
    cv2.rectangle(frame, (x, y), (x2, y2), (255, 0, 0), -1)

    # Προσθέτουμε και λίγα "features" στο background (σταθερά σημεία)
    for px in range(50, width, 100):
        for py in range(50, height, 100):
            cv2.circle(frame, (px, py), 3, (0, 0, 0), -1)

    out.write(frame)

out.release()
print(f"Saved synthetic video to {output_file}")
