import json
from pathlib import Path
from typing import Tuple

import numpy as np


class EthicalZoneManager:
    """
    Manages spatial "ethical zones", such as private areas or vulnerable regions.

    Zones are defined in a JSON file with structure:

    {
      "private_zones": [
        {"x": 10.0, "y": 5.0, "radius": 3.0}
      ],
      "vulnerable_zones": [
        {"x": 4.0, "y": 4.0, "radius": 2.0}
      ]
    }
    """

    def __init__(self, json_path: str | Path = "ethical_zones.json"):
        json_path = Path(json_path)
        if not json_path.exists():
            # No zones defined – default to empty.
            self.private_zones = []
            self.vulnerable_zones = []
            return

        with open(json_path, "r", encoding="utf-8") as f:
            zones = json.load(f)

        self.private_zones = zones.get("private_zones", [])
        self.vulnerable_zones = zones.get("vulnerable_zones", [])

    @staticmethod
    def _inside_circle(x: float, y: float, zone: dict) -> bool:
        dx = x - zone["x"]
        dy = y - zone["y"]
        return np.sqrt(dx * dx + dy * dy) <= zone["radius"]

    def check_zone_status(self, x: float, y: float) -> Tuple[bool, bool]:
        """
        Check whether (x, y) lies inside any private or vulnerable zone.

        Returns
        -------
        in_private, in_vulnerable : bool, bool
        """
        in_private = any(self._inside_circle(x, y, z) for z in self.private_zones)
        in_vulnerable = any(
            self._inside_circle(x, y, z) for z in self.vulnerable_zones
        )

        return in_private, in_vulnerable
