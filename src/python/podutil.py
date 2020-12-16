import math

from constants import Constants
from vec2 import ORIGIN, Vec2


def within(value: float, min: float, max: float) -> float:
    """
    Limit the given value to the given range
    """
    if value < min: return min
    if value > max: return max
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


class PodInfo:
    """
    The full internal state of a pod
    """
    def __init__(self, start_pos: Vec2):
        self.pos = start_pos
        self.vel = ORIGIN
        self.angle = 0.0
        self.nextCheckId = 0
        self.laps = 0

    def __str__(self):
        return "Pod pos=" + str(self.pos) +\
               " vel=" + str(self.vel) +\
               " angle=" + str(self.angle) +\
               " laps=" + str(self.laps)

class PlayInput:
    """
    All information provided to a controller at each turn
    """
    def __init__(self, pod: PodInfo, world):
        self.pos = pod.pos
        self.vel = pod.vel
        self.angle = pod.angle
        self.nextCheckId = pod.nextCheckId
        self.nextCheck = world.checkpoints[pod.nextCheckId]

class PlayOutput:
    """
    All information that a controller produces during a turn
    """
    def __init__(self):
        self.dir = ORIGIN
        self.thrust = 0

class Controller:
    """
    Base class for controllers
    """
    def play(self, pi: PlayInput) -> PlayOutput:
        raise TypeError("Abstract base class")

class SimpleController(Controller):
    """
    A simple controller to illustrate the concept
    """
    def play(self, pi: PlayInput) -> PlayOutput:
        out = PlayOutput()
        out.dir = pi.nextCheck - (pi.vel * 2.0)
        out.thrust = 100
        return out