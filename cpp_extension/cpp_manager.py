from coverage_grid import CoverageGrid
from frontier_selector import find_frontiers

class CPPManager:
    def __init__(self, occ_width, occ_height, resolution):
        self.cg = CoverageGrid(occ_width, occ_height, resolution)

    def update_pose(self, x, y):
        self.cg.mark_visited(x,y)

    def decide_next_goal(self, occupancy):
        frontiers = find_frontiers(occupancy, self.cg.coverage)

        if len(frontiers) == 0:
            return None
        
        # pick nearest frontier
        fx, fy = frontiers[0]
        return (fx, fy)
