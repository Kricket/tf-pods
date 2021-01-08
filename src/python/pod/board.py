from typing import List
import random

from pod.constants import Constants
from vec2 import Vec2


class PodBoard:
    def __init__(self, checks: List[Vec2] = None):
        if checks == None:
            self.__generate_checks()
        else:
            self.checkpoints = checks.copy()

    def __generate_checks(self):
        min_x = Constants.border_padding()
        min_y = Constants.border_padding()
        max_x = Constants.world_x() - Constants.border_padding()
        max_y = Constants.world_y() - Constants.border_padding()
        min_dist_sq = Constants.check_spacing() * Constants.check_spacing()
        self.checkpoints = []

        num_checks = random.randrange(Constants.min_checks(), Constants.max_checks())
        while len(self.checkpoints) < num_checks:
            check = Vec2(random.randrange(min_x, max_x, 1), random.randrange(min_y, max_y, 1))
            too_close = next(
                (True for x in self.checkpoints if (x - check).square_length() < min_dist_sq),
                False)
            if not too_close:
                self.checkpoints.append(check)
