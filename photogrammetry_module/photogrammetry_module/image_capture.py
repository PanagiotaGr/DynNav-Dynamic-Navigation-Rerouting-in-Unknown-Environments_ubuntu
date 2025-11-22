import os
from datetime import datetime

import cv2


class ImageCapture:
    """
    Simple image capture utility for photogrammetry missions.
    """

    def __init__(self,
                 device_id: int = 0,
                 output_dir: str = None):
        """
        :param device_id: /dev/video{device_id}
        :param output_dir: directory where images will be stored
        """
        self.device_id = device_id
        if output_dir is None:
            output_dir = os.path.join(
                os.path.dirname(__file__),
                "images"
            )
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def capture(self) -> str:
        """
        Capture a single frame and save it to disk.
        :return: path to saved image
        """
        cap = cv2.VideoCapture(self.device_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera device {self.device_id}")

        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError("Could not read frame from camera")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(self.output_dir, f"frame_{timestamp}.jpg")

        cv2.imwrite(filename, frame)
        print(f"[ImageCapture] Saved image: {filename}")
        return filename
