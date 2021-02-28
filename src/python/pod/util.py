import math

from pod.constants import Constants
from vec2 import Vec2, ORIGIN


def within(value: float, mini: float, maxi: float) -> float:
    """
    Limit the given value to the given range
    """
    if value < mini: return mini
    if value > maxi: return maxi
    return value


def clean_angle(angle: float) -> float:
    """
    :param angle: Angle in radians
    :return: Same angle, in the range [-pi, pi]
    """
    while angle < -math.pi:
        angle += 2 * math.pi

    while angle > math.pi:
        angle -= 2 * math.pi

    return angle


def legal_angle(req_angle: float, pod_angle: float) -> float:
    """
    Get the actual angle to apply, given the player's input
    :param req_angle: Angle that the player requested
    :param pod_angle: Angle in which the pod is facing
    :return: Angle to use for calculations (within [-pi, pi])
    """
    d_angle = within(
        clean_angle(req_angle - pod_angle),
        -Constants.max_turn(),
        Constants.max_turn())
    return clean_angle(pod_angle + d_angle)


class PodState:
    """
    The full internal state of a pod
    """
    def __init__(self, pos: Vec2 = ORIGIN, vel: Vec2 = ORIGIN, angle: float = 0.0, next_check_id: int = 0):
        self.pos = pos
        self.vel = vel
        self.angle = angle
        self.nextCheckId = next_check_id
        self.laps = 0
        self.turns = 0

    def __str__(self):
        return "PodState[pos=%s vel=%s angle=%.3f laps=%d]" % (self.pos, self.vel, self.angle, self.laps)

    def __eq__(self, other):
        if not isinstance(other, PodState):
            return False
        # NOTE: we ignore laps and turns here!
        if self.pos == other.pos:
            if self.vel == other.vel:
                if self.angle == other.angle:
                    if self.nextCheckId == other.nextCheckId:
                        return True
        return False

    def serialize(self):
        return [
            self.pos,
            self.vel,
            self.angle,
            self.nextCheckId,
            self.laps,
            self.turns
        ]

    def deserialize(self, state):
        self.pos = state[0]
        self.vel = state[1]
        self.angle = state[2]
        self.nextCheckId = state[3]
        self.laps = state[4]
        self.turns = state[5]

    def clone(self):
        other = PodState(self.pos, self.vel, self.angle, self.nextCheckId)
        other.laps = self.laps
        other.turns = self.turns
        return other
