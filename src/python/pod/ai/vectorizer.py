from typing import List

from pod.ai.ai_utils import MAX_DIST
from pod.board import PodBoard
from pod.constants import Constants
from pod.util import PodState


class Vectorizer:
    def to_vector(self, board: PodBoard, pod: PodState) -> List[float]:
        """
        Convert the state of a pod to a vector (to use as NN input)
        """
        pass
    def vec_len(self) -> int:
        """
        Length of the vector this class produces
        """
        pass

class V4(Vectorizer):
    """
    Converts to a 4-element vector: pod velocity, next check
    """
    def to_vector(self, board: PodBoard, pod: PodState) -> List[float]:
        # Velocity is already relative to the pod, so it just needs to be rotated
        vel = pod.vel.rotate(-pod.angle) / Constants.max_vel()

        check1 = (board.get_check(pod.nextCheckId) - pod.pos).rotate(-pod.angle) / MAX_DIST

        return [vel.x, vel.y, check1.x, check1.y]

    def vec_len(self) -> int:
        return 4


class V6(Vectorizer):
    """
    Converts to a 6-element vector: pod velocity, next check, following check
    """
    def to_vector(self, board: PodBoard, pod: PodState) -> List[float]:
        # Velocity is already relative to the pod, so it just needs to be rotated
        vel = pod.vel.rotate(-pod.angle) / Constants.max_vel()

        check1 = (board.get_check(pod.nextCheckId) - pod.pos).rotate(-pod.angle) / MAX_DIST
        check2 = (board.get_check(pod.nextCheckId + 1) - pod.pos).rotate(-pod.angle) / MAX_DIST

        return [vel.x, vel.y, check1.x, check1.y, check2.x, check2.y]

    def vec_len(self) -> int:
        return 6

