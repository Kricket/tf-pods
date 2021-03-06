from typing import Dict

from pod.controller import Controller
from pod.util import PodState


class Player:
    """
    A Player encapsulates both a Controller and its associated PodState
    """

    def __init__(self, controller: Controller, pod: PodState = None):
        """
        :param controller: Handles movement
        :param pod: Initial state of the pod. If omitted, a new one will be created
        """
        self.controller = controller
        if pod is None:
            self.pod = PodState(controller.board.checkpoints[-1])
        else:
            self.pod = pod

        self.initial_state = self.pod.clone()

    def __str__(self):
        return "Player[controller=%s pod=%s]" % (type(self.controller), self.pod)

    def step(self):
        """
        Have the Controller play once, and update the pod with the output
        """
        self.controller.step(self.pod)

    def reset(self):
        """
        Reset: put the pod at the start position with 0 turns/laps
        """
        self.pod = self.initial_state.clone()
        self.controller.reset()

    def record(self) -> Dict:
        log = {'pod': self.pod.clone()}
        self.controller.record(log)
        return log
