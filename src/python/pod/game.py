from pod.constants import Constants
from pod.board import PodBoard
from pod.controller import Controller, PlayInput, PlayOutput
from pod.util import legal_angle, within, PodState
from vec2 import UNIT, ORIGIN


class Player:
    def __init__(self, controller: Controller, pod: PodState = None, board: PodBoard = None):
        """
        Create a Player
        :param controller: Handles movement
        :param pod: Initial state of the pod. If omitted, a new one will be created
        :param board: If supplied and no pod is given, the created pod will start on the last checkpoint
        """
        self.controller = controller
        if pod is None:
            if board is None:
                self.pod = PodState(ORIGIN)
            else:
                self.pod = PodState(board.checkpoints[-1])
        else:
            self.pod = pod

    def __str__(self):
        return "Player[controller=%s pod=%s]" % (type(self.controller), self.pod)

    def step(self, board: PodBoard):
        """
        Have the Controller play once, and update the pod with the output
        :param board: The board on which to play
        """
        pi = PlayInput(self.pod, board)
        game_step(board, self.pod, self.controller.play(pi), self.pod)


def game_step(board: PodBoard, pod: PodState, play: PlayOutput, output: PodState):
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
    :param board: The board on which the game is being played
    :param pod: PodInfo starting state
    :param play: PlayInput to play
    :param output: PodInfo output state to update.
    :return: The new pod state
    """
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
    check = board.checkpoints[pod.nextCheckId]
    if (check - output.pos).square_length() < Constants.check_radius_sq():
        output.nextCheckId += 1
        if output.nextCheckId >= len(board.checkpoints):
            output.nextCheckId = 0
            output.laps += 1
