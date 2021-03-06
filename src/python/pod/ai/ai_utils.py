import math
from typing import List

import numpy as np
from pod.constants import Constants
from pod.controller import Controller
from pod.util import PodState, clean_angle
from vec2 import Vec2, UNIT

# Distance to use for scaling inputs
MAX_DIST = Vec2(Constants.world_x(), Constants.world_y()).length()


def gen_pods(
        checks: List[Vec2],
        pos_angles: List[float],
        pos_dists: List[float],
        angles: List[float],
        vel_angles: List[float],
        vel_mags: List[float]
):
    """
    Generate pods in various states
    :param checks: Checkpoints around which to generate
    :param pos_angles: Angles from check to pod
    :param pos_dists: Distances from check to pod
    :param angles: Orientations of pods. This will be rotated so that 0 points toward the check!
    :param vel_angles: Angles of velocity. Also rotated so that 0 points toward the check.
    :param vel_mags: Magnitudes of velocity
    :return: One pod for each combination of parameters
    """
    relative_poss = [UNIT.rotate(ang) * dist for ang in pos_angles for dist in pos_dists]
    relative_vels = [UNIT.rotate(ang) * mag for ang in vel_angles for mag in vel_mags]

    print("Generating pods: checks={} positions={} angles={} vels={}".format(
        len(checks), len(relative_poss), len(angles), len(relative_vels)
    ))

    pods = []

    for (c_idx, checkpoint) in enumerate(checks):
        for rel_pos in relative_poss:
            ang_to_check = rel_pos.angle() + math.pi
            pos = checkpoint + rel_pos
            for rel_vel in relative_vels:
                vel = rel_vel.rotate(ang_to_check)
                for angle in angles:
                    pods.append(PodState(
                        pos=pos,
                        vel=vel,
                        angle=clean_angle(angle + ang_to_check),
                        next_check_id=c_idx))

    np.random.shuffle(pods)
    print("{} pods generated".format(len(pods)))
    return pods


def play_gen_pods(start_pods: List[PodState],
                  controller: Controller,
                  turns: int) -> List[PodState]:
    """
    Play each given pod a few turns, to generate even more states.
    This will modify the input list of pods!
    """
    all_pods = []
    for pod in start_pods:
        all_pods.append(pod.clone())
        for turn in range(turns):
            controller.step(pod)
            all_pods.append(pod.clone())

    np.random.shuffle(all_pods)
    print("{} pods generated".format(len(all_pods)))
    return all_pods
