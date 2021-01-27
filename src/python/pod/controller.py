from pod.board import PodBoard
from pod.util import PodState
from vec2 import ORIGIN


class PlayInput:
    """
    All information provided to a controller at each turn
    """
    def __init__(self, pod: PodState, board: PodBoard):
        self.pos = pod.pos
        self.vel = pod.vel
        self.angle = pod.angle
        self.nextCheckId = pod.nextCheckId
        self.nextCheck = board.checkpoints[pod.nextCheckId]

    def as_pod(self) -> PodState:
        """
        Get this as a PodState (some data may be missing...)
        """
        pod = PodState(self.pos)
        pod.vel = self.vel
        pod.angle = self.angle
        pod.nextCheckId = self.nextCheckId
        return pod

    def __str__(self):
        return "PI[pos={} vel={} angle={} nextId={}]".format(self.pos, self.vel, self.angle, self.nextCheckId)


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


class Controller:
    """
    Base class for a Controller: a thing that produces a play for a given input
    """
    def play(self, pi: PlayInput) -> PlayOutput:
        return PlayOutput()


class SimpleController(Controller):
    """
    A simple Controller that just goes full-speed toward the next checkpoint
    """
    def play(self, pi: PlayInput) -> PlayOutput:
        out = PlayOutput()
        out.target = pi.nextCheck - (pi.vel * 2.0)
        out.thrust = 100
        return out
