import math
from typing import Dict

from pod.board import PodBoard, PlayOutput
from pod.constants import Constants
from pod.util import PodState, clean_angle


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

    def reset(self):
        """
        Do anything that needs to be done to reset the internal state
        """
        pass

    def record(self, log: Dict):
        """
        Save data about the current state of this Controller for future use
        """
        pass


class SimpleController(Controller):
    """
    A simple Controller that just goes full-speed toward the next checkpoint
    """
    def play(self, pod: PodState) -> PlayOutput:
        out = PlayOutput()
        out.target = self.board.checkpoints[pod.nextCheckId]
        out.thrust = 100
        return out

class MediumController(Controller):
    """
    A slightly more clever controller, but still nothing great
    """
    def play(self, pod: PodState) -> PlayOutput:
        out = PlayOutput()
        check = self.board.checkpoints[pod.nextCheckId]
        out.target = check - pod.vel

        ang_to_check = (check - pod.pos).angle()
        if math.fabs(clean_angle(pod.angle - ang_to_check)) < math.pi / 2:
            out.thrust = 100
        else:
            out.thrust = 0

        return out

class CleverController(Controller):
    """
    My best shot at a manual implementation
    """
    def play(self, pod: PodState) -> PlayOutput:
        check1 = self.board.checkpoints[pod.nextCheckId]
        check2 = self.board.get_check(pod.nextCheckId + 1)
        c1_to_p = (pod.pos - check1)
        c1_to_p_len = c1_to_p.length()
        c1_to_c2 = (check2 - check1)
        c1_to_c2_len = c1_to_c2.length()

        midpoint = ((c1_to_p / c1_to_c2_len) - (c1_to_c2 / c1_to_c2_len)).normalize()
        target = check1

        if c1_to_p_len > Constants.max_vel() * 6:
            # Still far away. Aim for a point that will help us turn toward the next check
            target = target + (midpoint * Constants.check_radius() * 2)
        # else: We're getting close to the check. Stop fooling around and go to it.

        # OK, now we've got a target point. Do whatever it takes to get there.
        pod_to_target = target - pod.pos
        ang_diff_to_target = math.fabs(clean_angle(math.fabs(pod.angle - pod_to_target.angle())))

        if ang_diff_to_target < 2 * Constants.max_turn():
            thrust = Constants.max_thrust()
        elif ang_diff_to_target < 4 * Constants.max_turn():
            thrust = (ang_diff_to_target - (4 * Constants.max_turn())) / (2 * Constants.max_turn()) * Constants.max_thrust()
        else:
            thrust = 0

        return PlayOutput(target - (2*pod.vel), thrust)

