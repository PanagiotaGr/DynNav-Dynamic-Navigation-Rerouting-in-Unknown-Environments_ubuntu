import numpy as np

class CoverageGrid:
    def __init__(self, width, height, resolution):
        self.resolution = resolution
        self.coverage = np.zeros((height, width), dtype=np.uint8)

    def mark_visited(self, x, y):
        i = int(y / self.resolution)
        j = int(x / self.resolution)
        if 0 <= i < self.coverage.shape[0] and 0 <= j < self.coverage.shape[1]:
            self.coverage[i, j] = 1

    def coverage_ratio(self):
        total = self.coverage.size
        visited = np.count_nonzero(self.coverage)
        return visited / total
