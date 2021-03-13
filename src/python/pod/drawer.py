import math
from pathlib import Path
from typing import List, Tuple, Callable, Dict

import matplotlib.pyplot as plt
import matplotlib.rcsetup
from IPython.display import Image, HTML
from matplotlib.animation import FuncAnimation, PillowWriter, HTMLWriter
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Wedge, Rectangle

from logger import JupyterLog
from pod.board import PodBoard
from pod.constants import Constants
from pod.controller import Controller
from pod.player import Player
from pod.util import PodState
from vec2 import Vec2

# extra space around the edge of the actual game area to show
PADDING = 3000


def _prepare_size():
    plt.rcParams['figure.figsize'] = [Constants.world_x() / 1000, Constants.world_y() / 1000]
    plt.rcParams['figure.dpi'] = 100
    matplotlib.rcParams['animation.embed_limit'] = 2**27


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


def _to_line(a: Vec2, b: Vec2) -> List[Tuple[float, float]]:
    return [(a.x, a.y), (b.x, b.y)]


def _vel_coords(pod: PodState) -> Tuple[List[float], List[float]]:
    """
    Get the coordinates of a line segment to use to show the velocity of the given pod
    """
    return (
        # X-coordinates
        [pod.pos.x, pod.pos.x + pod.vel.x * 3],
        # Y-coordinates
        [pod.pos.y, pod.pos.y + pod.vel.y * 3]
    )


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

        self.hist: List[List[Dict]] = []

    def record(self, max_frames: int = 200, max_laps: int = 5, reset: bool = True):
        """
        Record and cache a run through the game with the current players
        :param max_frames: Maximum number of turns
        :param max_laps: Maximum number of laps
        :param reset: Whether to reset the players
        """
        self.__reset_if(reset)
        self.hist = [[p.record() for p in self.players]]

        log = JupyterLog()

        while max(p.pod.laps for p in self.players) < max_laps and len(self.hist) < max_frames:
            log.replace("Playing turn {}".format(len(self.hist)))
            turnlog = []
            for p in self.players:
                p.step()
                turnlog.append(p.record())
            self.hist.append(turnlog)

    def __prepare_for_world(self):
        _prepare_size()
        self.fig = plt.figure()
        self.ax = plt.axes(
            xlim=(-PADDING, Constants.world_x() + PADDING),
            ylim=(-PADDING, Constants.world_y() + PADDING)
        )
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
        log = JupyterLog()

        frames = []
        while max(p.pod.laps for p in self.players) < max_laps and len(frames) < max_frames:
            log.replace("Generating frame {}".format(len(frames)))
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
                highlight_checks: bool = True,
                show_vel: bool = True,
                fps: int = 10):
        """
        Generate an animated GIF of the players running through the game
        :param show_vel Whether to draw a vector showing each Pod's velocity
        :param highlight_checks If true, a pod's next check will change to the pod's color
        :param trail_len Number of turns behind the pod to show its path
        :param as_gif If True, generate a GIF, otherwise an HTML animation
        :param max_frames Max number of turns to play
        :param max_laps Max number of laps for any player
        :param filename Where to store the generated file
        :param reset Whether to reset the state of each Player first
        :param fps Frames per second
        """
        if len(self.hist) < 1: self.record(max_frames, max_laps, reset)

        self.__prepare_for_world()

        #########################################
        # Create the objects for display
        #########################################

        art = {
            'check': [],
            'pod': [],
            'color': [],
            'trails': [],
            'vel': [],
            'count': 0,
            'log': JupyterLog()
        }

        fa = _get_field_artist()
        self.ax.add_artist(fa)

        for (idx, check) in enumerate(self.board.checkpoints):
            ca = self.__draw_check(check, idx)
            self.ax.add_artist(ca)
            art['check'].append(ca)

        for (idx, p) in enumerate(self.players):
            color = _gen_color(idx)
            pa = _get_pod_artist(p.pod, color)
            self.ax.add_artist(pa)
            art['pod'].append(pa)
            art['color'].append(color)
        plt.legend(art['pod'], self.labels)

        if trail_len > 0:
            for i in range(len(self.players)):
                lc = LineCollection([], colors = art['color'][i])
                lc.set_segments([])
                lc.set_linestyle(':')
                self.ax.add_collection(lc)
                art['trails'].append(lc)

        if show_vel:
            for p in self.players:
                xy = _vel_coords(p.pod)
                line = self.ax.plot(xy[0], xy[1])[0]
                art['vel'].append(line)

        all_updates = [
            fa, *art['check'], *art['pod'], *art['trails'], *art['vel']
        ]

        #########################################
        # Define the animation function
        #########################################

        def do_animate(frame_idx: int):
            art['log'].replace("Drawing frame {}".format(art['count']))
            art['count'] += 1

            check_colors = ['royalblue' for _ in range(len(self.board.checkpoints))]
            frame_data = self.hist[frame_idx]

            # Update the pods
            for (p_idx, player_log) in enumerate(frame_data):
                pod = player_log['pod']
                theta1, theta2, center = _pod_wedge_info(pod)
                art['pod'][p_idx].set_center((center.x, center.y))
                art['pod'][p_idx].set_theta1(theta1)
                art['pod'][p_idx].set_theta2(theta2)
                art['pod'][p_idx]._recompute_path() # pylint: disable=protected-access

                check_colors[pod.nextCheckId] = art['color'][p_idx]

                # Update the velocities
                if show_vel:
                    xy = _vel_coords(pod)
                    art['vel'][p_idx].set_xdata(xy[0])
                    art['vel'][p_idx].set_ydata(xy[1])

            # Update the trails
            if frame_idx > 0 and trail_len > 0:
                for p_idx in range(len(self.players)):
                    line = _to_line(
                        self.hist[frame_idx-1][p_idx]['pod'].pos,
                        frame_data[p_idx]['pod'].pos
                    )
                    segs = art['trails'][p_idx].get_segments() + [line]
                    art['trails'][p_idx].set_segments(segs[-trail_len:])

            # Update the check colors
            if highlight_checks:
                for col, check_art in zip(check_colors, art['check']): check_art.set_color(col)

            return all_updates

        #########################################
        # Create the animation
        #########################################

        anim = FuncAnimation(
            plt.gcf(),
            do_animate,
            frames = len(self.hist),
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


    def chart_rewards(self,
                      reward_func: Callable[[PodBoard, PodState, PodState], float]):
        """
        Display a graph of the rewards for each player at each turn
        """
        if len(self.hist) < 1: self.record()
        _prepare_size()

        for (player_idx, player) in enumerate(self.players):
            rewards = []
            for frame_idx in range(1, len(self.hist)):
                prev_log = self.hist[frame_idx - 1][player_idx]
                next_log = self.hist[frame_idx][player_idx]
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

        if len(self.hist) < 1: self.record()
        _prepare_size()

        ticks = []
        tick_labels = []
        tick_colors = []
        for step in range(0, len(self.hist), 10):
            ticks.append(step)
            tick_labels.append(str(step))
            tick_colors.append('black')


        for (p_idx, player) in enumerate(players):
            for (r_idx, r_def) in enumerate(rewarders):
                player_color = _gen_color(r_idx * len(players) + p_idx)
                rewards = []
                for f_idx in range(1, len(self.hist)):
                    prev_pod = self.hist[f_idx-1][p_idx]['pod']
                    next_pod = self.hist[f_idx][p_idx]['pod']
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
