import random
from typing import List

from constants import Constants
from podutil import PodInfo, legal_angle, PlayOutput, Controller, within, PlayInput
from vec2 import Vec2, UNIT


class Player:
    def __init__(self, pod: PodInfo, controller: Controller):
        self.pod = pod
        self.controller = controller


class PodWorld:
    """
    The game. Stores the full state of the board and pods, and implements
    the physics and stepping functions.
    """
    def __init__(self):
        self.__generate_checks()
        self.players: List[Player] = []

    def __str__(self):
        checks = str(list(str(check) for check in self.checkpoints))
        pods = str(list(str(pc.pod) for pc in self.players))
        return checks + '\n' + pods

    def __generate_checks(self):
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

    def add_player(self, controller: Controller):
        pod = PodInfo(self.checkpoints[-1])
        self.players.append(Player(pod, controller))

    def step(self):
        """
        Move the simulation forward one turn
        """
        for pc in self.players:
            play_input = PlayInput(pc.pod, self)
            play = pc.controller.play(play_input)
            self.update(pc.pod, play, pc.pod)

    def update(self, pod: PodInfo, play: PlayOutput, output: PodInfo):
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
        :param pod: PodInfo starting state
        :param play: PlayInput to play
        :param output: PodInfo output state to update.
        :return: The new pod state
        """
        # 1. Rotation
        requested_angle = (play.dir - pod.pos).angle()
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
        output.nextCheckId = pod.nextCheckId
        output.laps = pod.laps
        check = self.checkpoints[pod.nextCheckId]
        if (check - output.pos).square_length() < Constants.check_radius_sq():
            output.nextCheckId += 1
            if output.nextCheckId >= len(self.checkpoints):
                output.nextCheckId = 0
                output.laps += 1
