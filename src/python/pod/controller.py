from pod.board import PodBoard, PlayOutput
from pod.util import PodState


class Controller:
    """
    Base class for a Controller: a thing that produces a play for a given input
    """
    def __init__(self, board: PodBoard):
        self.board = board

    def step(self, pod: PodState):
        """
        Move the given PodState forward one step in the simulation
        """
        self.board.step(pod, self.play(pod), pod)

    def play(self, pod: PodState) -> PlayOutput:
        """
        Get the play that this Controller will make
        """
        return PlayOutput()


class SimpleController(Controller):
    """
    A simple Controller that just goes full-speed toward the next checkpoint
    """
    def play(self, pod: PodState) -> PlayOutput:
        out = PlayOutput()
        out.target = self.board.checkpoints[pod.nextCheckId] - (pod.vel * 2.0)
        out.thrust = 100
        return out
