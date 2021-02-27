from typing import List, Tuple, Callable

import matplotlib.pyplot as plt
import math

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Wedge, Rectangle
from IPython.display import Image, display
from ipywidgets import IntProgress

from pod.constants import Constants
from pod.board import PodBoard
from pod.controller import Controller
from pod.player import Player
from pod.util import PodState
from vec2 import Vec2


# extra space around the edge of the actual game area to show
PADDING = 3000


def _gen_color(seed: int) -> Tuple[float, float, float]:
    """
    Generate a color based on the given seed
    :param seed: A small integer (i.e. the index of the pod)
    :return:
    """
    color = ((seed * 167) % 13) / 13
    return color, 1 - color, seed % 2


def _gen_label(idx: int, player: Player) -> str:
    return "Player {} ({})".format(idx, type(player.controller).__name__)


class Drawer:
    def __init__(self,
                 board: PodBoard,
                 players: List[Player] = None,
                 controllers: List[Controller] = None,
                 labels: List[str] = None):
        self.board = board
        if players is not None:
            self.players = players
        elif controllers is not None:
            self.players = [Player(c) for c in controllers]
        else:
            raise ValueError('Must provide either players or controllers')
        
        if labels is None:
            labels = [_gen_label(idx, player) for (idx, player) in enumerate(self.players)]
        elif len(labels) < len(self.players):
            for idx in range(len(labels), len(self.players)):
                labels.append(_gen_label(idx, self.players[idx]))
        self.labels = labels

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

    def draw_frame(self, pods = None):
        """
        Draw a single frame of the game in its current state (board, players)
        """
        self.__prepare()

        self.ax.add_artist(self.__get_field_artist())

        for (idx, check) in enumerate(self.board.checkpoints):
            circle = self.__draw_check(check, idx)
            self.ax.add_artist(circle)

        if pods is None:
            for (idx, player) in enumerate(self.players):
                self.ax.add_artist(self.__get_pod_artist(player.pod, _gen_color(idx)))
        else:
            for (idx, pod) in enumerate(pods):
                self.ax.add_artist(self.__get_pod_artist(pod, _gen_color(idx)))

        plt.show()

    def __get_frames(self, max_frames: int, max_laps: int):
        frames = []
        while max(p.pod.laps for p in self.players) < max_laps and len(frames) < max_frames:
            for p in self.players:
                p.step()
            states = map(lambda pl: self.__pod_wedge_info(pl.pod), self.players)
            frames.append(enumerate(list(states)))
        return frames


    def animate(self, max_frames: int = 200, max_laps: int = 5, filename = '/tmp/pods.gif', reset: bool = True):
        """
        Generate an animated GIF of the players running through the game
        :param max_frames Max number of turns to play
        :param max_laps Max number of laps for any player
        :param filename Where to store the generated file
        :param reset Whether to reset the state of each Player first
        """
        if reset:
            for p in self.players:
                p.reset()

        self.__prepare()

        back_artists = []
        def draw_background():
            back_artists.append(self.__get_field_artist())
            for (idx, check) in enumerate(self.board.checkpoints):
                back_artists.append(self.__draw_check(check, idx))

            for artist in back_artists:
                self.ax.add_artist(artist)

            return back_artists

        pod_artists = list(self.__get_pod_artist(p.pod, _gen_color(idx)) for (idx, p) in enumerate(self.players))
        for a in pod_artists: self.ax.add_artist(a)
        plt.legend(pod_artists, self.labels)

        frames = self.__get_frames(max_frames, max_laps)

        progress = IntProgress(min=0, max=len(frames))
        display(progress)

        def do_animate(framedata):
            for (idx, frame) in framedata:
                theta1, theta2, center, check_id = frame
                pod_artists[idx].set_center((center.x, center.y))
                pod_artists[idx].set_theta1(theta1)
                pod_artists[idx].set_theta2(theta2)
                pod_artists[idx]._recompute_path() # pylint: disable=protected-access
                back_artists[check_id + 1].set_color((1, 0, 0))
            progress.value += 1
            return pod_artists

        anim = FuncAnimation(plt.gcf(), do_animate, init_func = draw_background, interval = 300, frames = frames, blit = True)
        plt.close(self.fig)
        anim.save(filename, writer = PillowWriter(fps=10))
        return Image(filename = filename)


    def chart_rewards(self,
                      reward_func: Callable[[PodBoard, PodState, PodState], float],
                      max_frames: int = 100,
                      reset: bool = True):
        """
        Display a graph of the rewards for each player at each turn
        """
        if reset:
            for p in self.players:
                p.reset()

        for (idx, player) in enumerate(self.players):
            rewards = []

            for frame in range(max_frames):
                old_pod = player.pod.clone()
                player.step()
                rewards.append(reward_func(self.board, old_pod, player.pod))

            plt.plot(rewards,
                     color=_gen_color(idx),
                     label=self.labels[idx])

        plt.legend(loc="upper left")
        plt.ylabel('Reward')
        plt.xlabel('Turns')
        plt.grid(axis='y')
        plt.show()