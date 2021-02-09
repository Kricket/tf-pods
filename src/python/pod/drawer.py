from typing import List, Tuple

import matplotlib.pyplot as plt
import math

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Wedge, Rectangle
from IPython.display import Image

from pod.ai.rewards import dense_reward
from pod.constants import Constants
from pod.board import PodBoard
from pod.game import Player
from pod.util import PodState
from vec2 import Vec2


# extra space around the edge of the actual game area to show
PADDING = 3000


def gen_color(seed: int) -> Tuple[float, float, float]:
    """
    Generate a color based on the given seed
    :param seed: A small integer (i.e. the index of the pod)
    :return:
    """
    color = ((seed * 167) % 13) / 13
    return color, 1 - color, seed % 2


class Drawer:
    def __init__(self, board: PodBoard, players: List[Player]):
        self.board = board
        self.players = players

    def __prepare(self):
        plt.rcParams['figure.figsize'] = [Constants.world_x() / 1000, Constants.world_y() / 1000]
        plt.rcParams['figure.dpi'] = 100
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-PADDING, Constants.world_x() + PADDING), ylim=(-PADDING, Constants.world_y() + PADDING))
        self.ax.invert_yaxis()

    def __draw_check(self, check: Vec2, idx: int) -> Circle:
        self.ax.annotate(str(idx), xy=(check.x, check.y), fontsize=20, ha="center")
        return Circle((check.x, check.y), Constants.check_radius())

    def __pod_wedge_info(self, pod: PodState):
        angle_deg = math.degrees(pod.angle) + 180.0
        offset = Vec2(Constants.pod_radius() / 2, 0).rotate(math.radians(angle_deg))
        center = pod.pos - offset
        return angle_deg - 20, angle_deg + 20, center, pod.nextCheckId

    def __get_pod_artist(self, pod: PodState, color: Tuple[float, float, float]) -> Wedge:
        # Draw the wedge
        theta1, theta2, center, check_id = self.__pod_wedge_info(pod)
        wedge = Wedge((center.x, center.y), Constants.pod_radius(), theta1, theta2, color = color)
        wedge.set_zorder(10)
        return wedge

    def __get_field_artist(self) -> Rectangle:
        return Rectangle(
            (0, 0),
            Constants.world_x(), Constants.world_y(),
            ec="black", fc="white")

    def draw(self):
        """
        Draw a single frame of the game in its current state (board, players)
        """
        self.__prepare()

        self.ax.add_artist(self.__get_field_artist())

        for (idx, check) in enumerate(self.board.checkpoints):
            circle = self.__draw_check(check, idx)
            self.ax.add_artist(circle)

        for (idx, player) in enumerate(self.players):
            self.ax.add_artist(self.__get_pod_artist(player.pod, gen_color(idx)))

        plt.show()

    def __get_frames(self, max_frames: int, max_laps: int):
        frames = []
        while max(p.pod.laps for p in self.players) < max_laps and len(frames) < max_frames:
            for p in self.players:
                p.step(self.board)
            states = map(lambda pl: self.__pod_wedge_info(pl.pod), self.players)
            frames.append(enumerate(list(states)))
        return frames


    def animate(self, max_frames: int = 200, max_laps: int = 5, filename = '/tmp/pods.gif'):
        """
        Generate an animated GIF of the players running through the game
        :param max_frames: Max number of turns to play
        :param max_laps: Max number of laps for any player
        :param filename: Where to store the generated file
        """
        self.__prepare()

        back_artists = []
        def draw_background():
            back_artists.append(self.__get_field_artist())
            for (idx, check) in enumerate(self.board.checkpoints):
                back_artists.append(self.__draw_check(check, idx))

            for artist in back_artists:
                self.ax.add_artist(artist)

            return back_artists

        pod_artists = list(self.__get_pod_artist(p.pod, gen_color(idx)) for (idx, p) in enumerate(self.players))
        for a in pod_artists: self.ax.add_artist(a)
        frames = self.__get_frames(max_frames, max_laps)

        def do_animate(framedata):
            for (idx, frame) in framedata:
                theta1, theta2, center, check_id = frame
                pod_artists[idx].set_center((center.x, center.y))
                pod_artists[idx].set_theta1(theta1)
                pod_artists[idx].set_theta2(theta2)
                pod_artists[idx]._recompute_path() # pylint: disable=protected-access
                back_artists[check_id + 1].set_color((1, 0, 0))
            return pod_artists

        anim = FuncAnimation(plt.gcf(), do_animate, init_func = draw_background, interval = 300, frames = frames, blit = True)
        plt.legend(pod_artists, [
            "Player {} ({})".format(p, type(self.players[p].controller).__name__)
            for p in range(len(self.players))
        ])
        plt.close(self.fig)
        anim.save(filename, writer = PillowWriter(fps=10))
        return Image(filename = filename)


    def chart_rewards(self, max_frames: int = 100):
        """
        Display a graph of the rewards for each player at each turn
        """
        for (idx, player) in enumerate(self.players):
            rewards = [dense_reward(player.pod, self.board)]
            for frame in range(max_frames):
                player.step(self.board)
                rewards.append(dense_reward(player.pod, self.board))
            plt.plot(rewards,
                     color=gen_color(idx),
                     label="Player {} ({})".format(idx, type(player.controller).__name__))

        plt.legend(loc="upper left")
        plt.ylabel('Reward')
        plt.xlabel('Turns')
        plt.grid(axis='y')
        plt.show()