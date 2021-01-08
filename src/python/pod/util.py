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
    Return the given angle, adjusted to be in [-pi, pi]
    :param angle: Angle in radians
    :return: Same angle, in the legal range
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
    def __init__(self, start_pos: Vec2 = ORIGIN):
        self.pos = start_pos
        self.vel = ORIGIN
        self.angle = 0.0
        self.nextCheckId = 0
        self.laps = 0
        self.turns = 0

    def __str__(self):
        return "PodState[pos=%s vel=%s angle=%.3f laps=%d]" % (self.pos, self.vel, self.angle, self.laps)

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
