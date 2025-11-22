import os
from datetime import datetime

import cv2


class ImageCapture:
    """
    Simple image capture utility for photogrammetry missions.
    Captures a single frame from a video device and stores it on disk.
    """

    def __init__(self,
                 device_id: int = 0,
                 output_dir: str = None):
        """
        :param device_id: camera index (e.g. /dev/video0 → 0)
        :param output_dir: directory where images will be stored
        """
        self.device_id = device_id

        if output_dir is None:
            # Default: photogrammetry_module/images/
            base_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(base_dir, "images")

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def capture(self) -> str:
        """
        Capture a single frame and save it to disk.
        :return: full path to the saved image
        """
        cap = cv2.VideoCapture(self.device_id)
        if not cap.isOpened():
            raise RuntimeError(f"[ImageCapture] Could not open camera device {self.device_id}")

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            raise RuntimeError("[ImageCapture] Could not read frame from camera")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.output_dir, f"frame_{timestamp}.jpg")

        success = cv2.imwrite(filename, frame)
        if not success:
            raise RuntimeError(f"[ImageCapture] Failed to write image to {filename}")

        print(f"[ImageCapture] Saved image: {filename}")
        return filename
