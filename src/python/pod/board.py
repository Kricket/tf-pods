import math
from typing import List
import random

from pod.constants import Constants
from pod.util import PodState, legal_angle, within
from vec2 import Vec2, UNIT, ORIGIN


class PlayOutput:
    """
    All information that a controller produces during a turn
    """
    def __init__(self, target = ORIGIN, thrust = 0):
        # Point towards which we want to move
        self.target = target
        self.thrust = thrust

    def __str__(self):
        return "thrust {} target {}".format(self.thrust, self.target)


class PodBoard:
    @staticmethod
    def circle(num_points: int = 3, radius: float = 4000) -> 'PodBoard':
        """
        Generate a PodBoard with checkpoints arranged in a circle around the
        center of the board
        """
        center = Vec2(Constants.world_x() / 2, Constants.world_y() / 2)
        angle_diff = 2 * math.pi / num_points
        v = UNIT * radius
        checks = [center + v.rotate(i * angle_diff) for i in range(num_points)]
        return PodBoard(checks)


    @staticmethod
    def grid(rows: int = 3, cols: int = 3, x_spacing: int = 4000, y_spacing: int = 3000) -> 'PodBoard':
        """
        Generate a board with checks in grid form:
        1 2 3
        4 5 6
        7 8 9
        """
        checks = []

        x_center = Constants.world_x() / 2
        y_center = Constants.world_y() / 2

        # 5 rows: -2, -1, 0, 1, 2
        # 4 rows: -1.5, -0.5, 0.5, 1.5
        # 3 rows: -1, 0, 1
        # => start at -(r-1)/2
        row_start = (1 - rows)/2
        col_start = (1 - cols)/2

        for row in range(rows):
            y_off = (row_start + row) * y_spacing
            for col in range(cols):
                x_off = (col_start + col) * x_spacing
                checks.append(Vec2(x_center + x_off, y_center + y_off))

        return PodBoard(checks)

    @staticmethod
    def trainer(num_checks: int = 3) -> 'PodBoard':
        """
        Generate a board with the given number of checks.
        They are all in a row, but at varying distances.
        The goal is to use it with gen_pods to generate test data with varying distances to the next check.
        """
        checks = [Vec2(Constants.check_radius() * ((i+1)**2), Constants.world_y() / 2)
                  for i in range(num_checks)]

        # Shift the checks to center them
        width = checks[-1].x - checks[0].x
        x_start = (Constants.world_x() - width) / 2 - checks[0].x

        return PodBoard([check + Vec2(x_start, 0)  for check in checks])

    @staticmethod
    def tester() -> 'PodBoard':
        """
        Generate a board laid out to test as many situations as possible
        (start) -> 0 -> 1: straight line
        1 -> 2: 180° turn
        2 -> 3: 90° turn
        3 -> 4 -> 5 -> 6: curve around to the right
        6 -> 7 (start): curve to the left
        """
        checks = []
        start = Vec2(Constants.world_x() / 10, Constants.world_y() / 2)
        checks.append(start + Vec2(5000, 0)) # straight ahead
        checks.append(checks[-1] + Vec2(6000, 0)) # straight ahead

        checks.append(checks[-1] + Vec2(-3000, 0)) # straight back

        checks.append(checks[-1] + Vec2(0, 2500)) # turn 90°

        checks.append(checks[-1] + Vec2(-3000, 1500)) # curve around
        checks.append(checks[-1] + Vec2(-3000, -1500)) # curve around
        checks.append(checks[-1] + Vec2(0, -5500)) # curve around

        checks.append(start) # turn other way

        return PodBoard(checks)


    def __init__(self, checks: List[Vec2] = None):
        if checks is None:
            self.__generate_random_checks()
        else:
            self.checkpoints = checks.copy()

    def __generate_random_checks(self):
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

    def shuffle(self) -> 'PodBoard':
        """
        Randomize the order of the checks
        """
        random.shuffle(self.checkpoints)
        return self

    def reorder(self, order: List[int]) -> 'PodBoard':
        """
        Put the existing checks into the given order
        """
        checks = [self.checkpoints[i] for i in order]
        self.checkpoints = checks
        return self

    def get_check(self, check_id: int):
        return self.checkpoints[check_id % len(self.checkpoints)]

    def step(self, pod: PodState, play: PlayOutput, output: PodState = None) -> PodState:
        """
        For the given pod, implement the given play.
        On each turn the pods movements are computed this way:
            Rotation: the pod rotates to face the target point, with a maximum of 18 degrees (except for the 1rst round).
            Acceleration: the pod's facing vector is multiplied by the given thrust value. The result is added to the current speed vector.
            Movement: The speed vector is added to the position of the pod. If a collision would occur at this point, the pods rebound off each other.
            Friction: the current speed vector of each pod is multiplied by 0.85
            The speed's values are truncated and the position's values are rounded to the nearest integer.
        Collisions are elastic. The minimum impulse of a collision is 120.
        A boost is in fact an acceleration of 650.
        A shield multiplies the Pod mass by 10.
        The provided angle is absolute. 0° means facing EAST while 90° means facing SOUTH.
        :param pod: Initial state
        :param play: Action to play
        :param output: Output state to update (may be the same as input pod). If not given, a new one will be created.
        :return: The new pod state (same object as output if given)
        """
        if output is None:
            output = PodState()

        # 1. Rotation
        requested_angle = (play.target - pod.pos).angle()
        angle = legal_angle(requested_angle, pod.angle)
        output.angle = angle

        # 2. Acceleration
        dir = UNIT.rotate(angle)
        thrust = int(within(play.thrust, 0, Constants.max_thrust()))
        output.vel = pod.vel + (dir * thrust)

        # 3. Movement
        output.pos = pod.pos + output.vel

        # 4. Friction
        output.vel = output.vel * Constants.friction()

        # 5. Rounding
        output.pos = output.pos.round()
        output.vel = output.vel.truncate()

        # Update progress
        output.turns = pod.turns + 1
        output.nextCheckId = pod.nextCheckId
        output.laps = pod.laps
        check = self.checkpoints[pod.nextCheckId]
        if (check - output.pos).square_length() < Constants.check_radius_sq():
            output.nextCheckId += 1
            if output.nextCheckId >= len(self.checkpoints):
                output.nextCheckId = 0
                output.laps += 1

        return output
