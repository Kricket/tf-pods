from pathlib import Path
from typing import List, Tuple, Callable, Dict

import matplotlib.pyplot as plt
import math
import numpy as np

from matplotlib.animation import FuncAnimation, PillowWriter, HTMLWriter
from matplotlib.patches import Circle, Wedge, Rectangle
from matplotlib.collections import LineCollection
from IPython.display import Image, display, HTML
from ipywidgets import widgets

from pod.constants import Constants
from pod.board import PodBoard
from pod.controller import Controller
from pod.player import Player
from pod.util import PodState
from vec2 import Vec2


# extra space around the edge of the actual game area to show
PADDING = 3000


def _prepare_size():
    plt.rcParams['figure.figsize'] = [Constants.world_x() / 1000, Constants.world_y() / 1000]
    plt.rcParams['figure.dpi'] = 100


def _get_field_artist() -> Rectangle:
    """
    Get an artist to draw the board
    """
    return Rectangle(
        (0, 0),
        Constants.world_x(), Constants.world_y(),
        ec="black", fc="white")


def _get_pod_artist(pod: PodState, color: Tuple[float, float, float]) -> Wedge:
    # Draw the wedge
    theta1, theta2, center = _pod_wedge_info(pod)
    wedge = Wedge((center.x, center.y), Constants.pod_radius(), theta1, theta2, color = color)
    wedge.set_zorder(10)
    return wedge


def _pod_wedge_info(pod: PodState) -> Tuple[float, float, Vec2]:
    """
    Get info for drawing a wedge for ta pod:
    angle from, angle to, center
    """
    angle_deg = math.degrees(pod.angle) + 180.0
    offset = Vec2(Constants.pod_radius() / 2, 0).rotate(math.radians(angle_deg))
    center = pod.pos - offset
    return angle_deg - 20, angle_deg + 20, center


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


def _gen_labels(players: List[Player]) -> List[str]:
    return [_gen_label(idx, player) for (idx, player) in enumerate(players)]


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
            labels = _gen_labels(self.players)
        elif len(labels) < len(self.players):
            for idx in range(len(labels), len(self.players)):
                labels.append(_gen_label(idx, self.players[idx]))
        self.labels = labels

        self.log = None

    def record(self, max_frames: int = 200, max_laps: int = 5, reset: bool = True):
        """
        Record and cache a run through the game with the current players
        :param max_frames: Maximum number of turns
        :param max_laps: Maximum number of laps
        :param reset: Whether to reset the players
        """
        self.__reset_if(reset)
        self.log = [[p.record() for p in self.players]]

        label = widgets.Label()
        display(label)

        while max(p.pod.laps for p in self.players) < max_laps and len(self.log) < max_frames:
            label.value = "Playing turn {}".format(len(self.log))
            turnlog = []
            for p in self.players:
                p.step()
                turnlog.append(p.record())
            self.log.append(turnlog)

    def __prepare_for_world(self):
        _prepare_size()
        self.fig = plt.figure()
        self.ax = plt.axes(xlim=(-PADDING, Constants.world_x() + PADDING), ylim=(-PADDING, Constants.world_y() + PADDING))
        self.ax.invert_yaxis()

    def __reset_if(self, do_it: bool):
        if do_it:
            for p in self.players:
                p.reset()

    def __draw_check(self, check: Vec2, idx: int) -> Circle:
        self.ax.annotate(str(idx), xy=(check.x, check.y), fontsize=20, ha="center")
        return Circle((check.x, check.y), Constants.check_radius())

    def draw_frame(self, pods = None):
        """
        Draw a single frame of the game in its current state (board, players)
        """
        self.__prepare_for_world()

        self.ax.add_artist(_get_field_artist())

        for (idx, check) in enumerate(self.board.checkpoints):
            circle = self.__draw_check(check, idx)
            self.ax.add_artist(circle)

        if pods is None:
            for (idx, player) in enumerate(self.players):
                self.ax.add_artist(_get_pod_artist(player.pod, _gen_color(idx)))
        else:
            for (idx, pod) in enumerate(pods):
                self.ax.add_artist(_get_pod_artist(pod, _gen_color(idx)))

        plt.show()

    def __get_frames(self, max_frames: int, max_laps: int):
        log = widgets.Label()
        display(log)

        frames = []
        while max(p.pod.laps for p in self.players) < max_laps and len(frames) < max_frames:
            log.value = "Generating frame {}".format(len(frames))
            for p in self.players:
                p.step()
            states = map(lambda pl: _pod_wedge_info(pl.pod), self.players)
            frames.append(enumerate(list(states)))
        return frames

    def animate(self,
                max_frames: int = 200,
                max_laps: int = 5,
                as_gif = False,
                filename = '/tmp/pods',
                reset: bool = True,
                trail_len: int = 20,
                fps: int = 10):
        """
        Generate an animated GIF of the players running through the game
        :param as_gif If True, generate a GIF, otherwise an HTML animation
        :param max_frames Max number of turns to play
        :param max_laps Max number of laps for any player
        :param filename Where to store the generated file
        :param reset Whether to reset the state of each Player first
        :param fps Frames per second
        """
        if self.log is None: self.record(max_frames, max_laps, reset)

        self.__prepare_for_world()

        check_artists = []
        for (idx, check) in enumerate(self.board.checkpoints):
            ca = self.__draw_check(check, idx)
            self.ax.add_artist(ca)
            check_artists.append(ca)

        pod_artists = [
            _get_pod_artist(p.pod, _gen_color(idx))
            for (idx, p) in enumerate(self.players)
        ]
        for a in pod_artists: self.ax.add_artist(a)
        plt.legend(pod_artists, self.labels)

        pod_trails = [LineCollection([], colors = _gen_color(i)) for i in range(len(self.players))]
        for p in pod_trails:
            p.set_segments([])
            p.set_linestyle(':')
            self.ax.add_collection(p)

        label = widgets.Label()
        display(label)
        c = [0]

        def draw_background():
            return [_get_field_artist()]

        prev_pos_log = [None]
        def do_animate(frame_log: List[Dict]):
            label.value = "Drawing frame {}".format(c[0])
            c[0] += 1
            check_colors = ['royalblue' for i in range(len(self.board.checkpoints))]
            pos_log = []
            for (idx, player_log) in enumerate(frame_log):
                pod = player_log['pod']
                theta1, theta2, center = _pod_wedge_info(pod)
                pod_artists[idx].set_center((center.x, center.y))
                pod_artists[idx].set_theta1(theta1)
                pod_artists[idx].set_theta2(theta2)
                pod_artists[idx]._recompute_path() # pylint: disable=protected-access
                pos_log.append((pod.pos.x, pod.pos.y))
                check_colors[pod.nextCheckId] = _gen_color(idx)

            if prev_pos_log[0] is not None and trail_len > 0:
                for idx in range(len(frame_log)):
                    line = [prev_pos_log[0][idx], pos_log[idx]]
                    segs = pod_trails[idx].get_segments() + [line]
                    pod_trails[idx].set_segments(segs[-trail_len:])

            prev_pos_log[0] = pos_log
            for col, check in zip(check_colors, check_artists): check.set_color(col)
            return pod_artists + check_artists + pod_trails

        anim = FuncAnimation(
            plt.gcf(),
            do_animate,
            init_func = draw_background,
            frames = self.log,
            blit = True
        )
        plt.close(self.fig)

        if as_gif:
            if not filename.endswith(".gif"): filename = filename + ".gif"
            anim.save(filename, writer=PillowWriter(fps=fps))
            return Image(filename=filename)
        else:
            if not filename.endswith(".html"): filename = filename + ".html"
            anim.save(filename, writer=HTMLWriter(fps=fps, embed_frames=True, default_mode='loop'))
            path = Path(filename)
            return HTML(path.read_text())


    def chart_rewards(self, reward_func: Callable[[PodBoard, PodState, PodState], float]):
        """
        Display a graph of the rewards for each player at each turn
        """
        if self.log is None: self.record()
        _prepare_size()


        for (player_idx, player) in enumerate(self.players):
            rewards = []
            for frame_idx in range(1, len(self.log)):
                prev_log = self.log[frame_idx - 1][player_idx]
                next_log = self.log[frame_idx][player_idx]
                rewards.append(reward_func(self.board, prev_log['pod'], next_log['pod']))

            plt.plot(rewards,
                     color=_gen_color(player_idx),
                     label=self.labels[player_idx])

        plt.legend(loc="upper left")
        plt.ylabel('Reward')
        plt.xlabel('Turns')
        plt.grid(axis='y')
        plt.show()


    def compare_rewards(self,
                        rewarders: List[Tuple[str, Callable[[PodBoard, PodState, PodState], float]]],
                        players: List[int] = None):
        """
        Compare the reward function for the given players
        :param rewarders: Reward functions to graph, with their labels
        :param players: Indices of the players to graph
        """

        if players is None:
            players = [i for i in range(len(self.players))]

        if self.log is None: self.record()
        _prepare_size()

        ticks = []
        tick_labels = []
        tick_colors = []
        for step in range(0, len(self.log), 10):
            ticks.append(step)
            tick_labels.append(str(step))
            tick_colors.append('black')


        for (p_idx, player) in enumerate(players):
            for (r_idx, r_def) in enumerate(rewarders):
                player_color = _gen_color(r_idx * len(players) + p_idx)
                rewards = []
                for f_idx in range(1, len(self.log)):
                    prev_pod = self.log[f_idx-1][p_idx]['pod']
                    next_pod = self.log[f_idx][p_idx]['pod']
                    rewards.append(r_def[1](self.board, prev_pod, next_pod))

                    if prev_pod.nextCheckId != next_pod.nextCheckId:
                        ticks.append(f_idx)
                        tick_labels.append("Check {}".format(prev_pod.nextCheckId))
                        tick_colors.append(player_color)
                plt.plot(rewards,
                     color=player_color,
                     label="{} - {}".format(self.labels[p_idx], r_def[0]))

        plt.legend(loc="lower right")
        plt.ylabel('Reward')
        plt.xlabel('Turns')
        locs, labels = plt.xticks(ticks=ticks, labels=tick_labels, rotation=60)
        for (i, lab) in enumerate(labels):
            lab.set_color(tick_colors[i])
        plt.grid()
        plt.show()
