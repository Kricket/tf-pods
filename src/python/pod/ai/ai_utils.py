import math
from typing import List

import numpy as np
from pod.constants import Constants
from pod.util import PodState, clean_angle
from vec2 import Vec2, UNIT

# Distance to use for scaling inputs
MAX_DIST = Vec2(Constants.world_x(), Constants.world_y()).length()


def gen_pods(
        checkpoint: Vec2,
        xy_list: List[int],
        ang_list: List[float],
        vel_ang_list: List[float],
        vel_mag_list: List[float]
):
    pods = []

    for x in xy_list:
        for y in xy_list:
            abs_pos = Vec2(x, y)
            ang_to_check = abs_pos.angle() + math.pi
            pos = abs_pos + checkpoint
            for a in ang_list:
                angle = clean_angle(ang_to_check + a)
                for va in vel_ang_list:
                    vel_dir = UNIT.rotate(ang_to_check + va)
                    for vm in vel_mag_list:
                        vel = vel_dir * vm

                        pod = PodState(pos)
                        pod.angle = angle
                        pod.vel = vel
                        pods.append(pod)

    np.random.shuffle(pods)
    print("{} pods generated".format(len(pods)))
    return pods
