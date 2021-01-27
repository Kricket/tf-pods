from typing import List, Tuple

import matplotlib.pyplot as plt
import math

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Wedge

from pod.ai.ai_utils import reward
from pod.constants import Constants
from pod.board import PodBoard
from pod.game import Player, game_step
from pod.util import PodState
from vec2 import Vec2


PADDING = 5000

def gen_color(seed: int) -> Tuple[float, float, float]:
    """
    Generate a color based on the given seed
    :param seed: A small integer (i.e. the number of the pod
    :return:
    """
    color = (seed * 12345 % 6789) / 6789.0
    return color, 1 - color, 0.0


class Drawer:
    def __init__(self, board: PodBoard, players: List[Player]):
        self.board = board
        self.players = players

    def __prepare(self):
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-PADDING, Constants.world_x() + PADDING), ylim=(-PADDING, Constants.world_y() + PADDING))
        self.ax.invert_yaxis()
        plt.rcParams['figure.figsize'] = [Constants.world_x() / 1000, Constants.world_y() / 1000]
        plt.rcParams['figure.dpi'] = 100

    def __draw_check(self, check: Vec2, idx: int) -> Circle:
        self.ax.annotate(str(idx), xy=(check.x, check.y), fontsize=20, ha="center")
        return Circle((check.x, check.y), Constants.check_radius())

    def __pod_wedge_info(self, pod: PodState):
        angle_deg = math.degrees(pod.angle) + 180.0
        offset = Vec2(Constants.pod_radius() / 2, 0).rotate(math.radians(angle_deg))
        center = pod.pos - offset
        return angle_deg - 20, angle_deg + 20, center, pod.nextCheckId

    def __draw_pod(self, pod: PodState, color: Tuple[float, float, float]) -> Wedge:
        # Draw the wedge
        theta1, theta2, center, check_id = self.__pod_wedge_info(pod)
        wedge = Wedge((center.x, center.y), Constants.pod_radius(), theta1, theta2, color = color)
        wedge.set_zorder(10)
        return wedge

    def draw(self):
        """
        Draw a single frame of the game in its current state (board, players)
        """
        self.__prepare()

        for (idx, check) in enumerate(self.board.checkpoints):
            circle = self.__draw_check(check, idx)
            self.ax.add_artist(circle)

        for (idx, player) in enumerate(self.players):
            self.ax.add_artist(self.__draw_pod(player.pod, gen_color(idx)))

        plt.show()

    def __get_frames(self, max_frames: int, max_laps: int):
        frames = []
        while max(p.pod.laps for p in self.players) < max_laps and len(frames) < max_frames:
            for p in self.players:
                p.step(self.board)
            states = map(lambda pl: self.__pod_wedge_info(pl.pod), self.players)
            frames.append(enumerate(list(states)))
        return frames


    def animate(self, filename, max_frames: int = 200, max_laps: int = 3):
        """
        Generate an animated GIF of the players running through the game
        :param filename: Where to store the generated file
        :param max_frames: Max number of turns to play
        :param max_laps: Max number of laps for any player
        """
        self.__prepare()

        checks = []
        def draw_checks():
            for (idx, check) in enumerate(self.board.checkpoints):
                circle = self.__draw_check(check, idx)
                self.ax.add_artist(circle)
                checks.append(circle)
            return checks

        artists = list(self.__draw_pod(p.pod, gen_color(idx)) for (idx, p) in enumerate(self.players))
        for a in artists: self.ax.add_artist(a)
        frames = self.__get_frames(max_frames, max_laps)

        def do_animate(framedata):
            for (idx, frame) in framedata:
                theta1, theta2, center, check_id = frame
                artists[idx].set_center((center.x, center.y))
                artists[idx].set_theta1(theta1)
                artists[idx].set_theta2(theta2)
                artists[idx]._recompute_path()
                checks[check_id].set_color((1, 0, 0))
            return artists

        anim = FuncAnimation(plt.gcf(), do_animate, init_func = draw_checks, interval = 300, frames = frames, blit = True)
        plt.legend(artists, [
            "Player {} ({})".format(p, type(self.players[p].controller).__name__)
            for p in range(len(self.players))
        ])
        plt.close(self.fig)
        anim.save(filename, writer = PillowWriter(fps=10))


    def chart_rewards(self, max_frames: int = 100):
        """
        Display a graph of the rewards for each player at each turn
        """
        rewards = [[reward(p.pod, self.board)] for p in self.players]
        for frame in range(max_frames):
            for p in range(len(self.players)):
                self.players[p].step(self.board)
                rewards[p].append(reward(self.players[p].pod, self.board))

        for (idx, r) in enumerate(rewards):
            plt.plot(r, color=gen_color(idx))

        plt.legend([
            "Player {} ({})".format(p, type(self.players[p].controller).__name__)
            for p in range(len(self.players))
        ])
        plt.ylabel('reward')
        plt.xlabel('player')
        plt.grid(axis='y')
        plt.show()