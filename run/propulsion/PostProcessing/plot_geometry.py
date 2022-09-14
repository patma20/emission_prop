#!/usr/bin/env python
"""
@File    :   plot_geometry.py
@Time    :   2022/04/20
@Desc    :   None
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import animation
from pyoptsparse import History

# ==============================================================================
# Extension modules
# ==============================================================================


def plot_HX(dir, fname, h_c=3.0, w_c=0.9, l_c=6.0, t_c=0.102):

    rows = 10
    x = np.array([])
    y = np.array([])
    x_an = np.linspace(0, (rows - 1) * w_c, rows)
    y_an = np.zeros(rows)
    for n in range(0, rows + 1, 1):
        x_seq = np.array([n * w_c, n * w_c])
        if (n % 2) == 0:
            y_seq = np.array([0, h_c])
        else:
            y_seq = np.array([h_c, 0])
        x = np.concatenate((x, x_seq))
        y = np.concatenate((y, y_seq))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    for i in range(0, rows):
        if (i % 2) == 0:
            color = "silver"
        else:
            color = "grey"
        ax1.add_patch(Rectangle((x_an[i], y_an[i]), width=w_c, height=l_c, edgecolor=color, facecolor=color, fill=True))
    ax2.plot(x, y, linewidth=5, color="grey")
    ax1.set_ylabel("fin length, [mm]")
    ax2.set_ylabel("channel height, [mm]")
    ax1.set_ylim([0, l_c])
    # ax2.set_ylim([0, h_c])
    # ax1.set_xlim([0, rows * w_c])
    # ax2.set_xlim([0, rows * w_c])
    ax1.set_aspect(aspect="equal")
    ax2.set_aspect(aspect="equal")
    ax2.set_xlabel("channel width, [mm]")
    asp = np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0]
    asp /= np.abs(np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0])
    ax1.set_aspect(asp)
    ax1.set_title("Top View", loc="right")
    ax2.set_title("Front View", loc="right")
    fig.tight_layout()
    fig.savefig(dir + fname + ".pdf")
    # plt.show()


class PlotGeo:
    def __init__(self, h, w, length, rows, dir, fname):
        self.h = h
        self.w = w
        self.length = length
        self.rows = rows
        self.dir = dir
        self.fname = fname
        self.xscale = 35  # self.rows * self.w[0]

    def plot_animation(self):
        # fig3 = plt.figure(figsize=(8, 8))
        # ax = fig3.add_subplot(111, projection="3d")
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 4))

        def Animation(j):
            h_c = self.h[j]
            w_c = self.w[j]
            l_c = self.length[j]
            ax1.clear()
            ax2.clear()

            x = np.array([])
            y = np.array([])
            x_an = np.linspace(0, (self.rows - 1) * w_c, self.rows)
            y_an = np.zeros(self.rows)
            for n in range(0, self.rows + 1, 1):
                x_seq = np.array([n * w_c, n * w_c])
                if (n % 2) == 0:
                    y_seq = np.array([0, h_c])
                else:
                    y_seq = np.array([h_c, 0])
                x = np.concatenate((x, x_seq))
                y = np.concatenate((y, y_seq))

            for i in range(0, self.rows):
                if (i % 2) == 0:
                    color = "silver"
                else:
                    color = "grey"
                ax1.add_patch(
                    Rectangle((x_an[i], y_an[i]), width=w_c, height=l_c, edgecolor=color, facecolor=color, fill=True)
                )
            ax2.plot(x, y, linewidth=5, color="grey")
            ax1.set_ylabel("fin length, [mm]")
            ax2.set_ylabel("channel height, [mm]")
            ax1.set_ylim([0, length[0]])
            ax2.set_ylim([0, self.h[0]])
            # if self.rows * w_c < 0.8 * self.xscale:
            #     self.xscale = self.rows * w_c
            ax1.set_xlim([0, self.xscale])
            ax2.set_xlim([0, self.xscale])
            ax1.set_aspect(aspect="equal")
            ax2.set_aspect(aspect="equal")
            # ax1.set_xlabel("width, [mm]")
            ax2.set_xlabel("channel width, [mm]")
            asp = np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
            asp /= np.abs(np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0])
            ax2.set_aspect(asp)
            fig.suptitle(f"Iteration: {j}")
            ax1.set_title("Top View", loc="right")
            ax2.set_title("Front View", loc="right")
            fig.tight_layout()
            # ax.legend(loc="upper left", fontsize=14)

        anim = animation.FuncAnimation(fig, func=Animation, frames=self.h.size, interval=200)
        plt.show()

        self.fname = self.dir + self.fname + ".gif"
        writergif = animation.PillowWriter(fps=6)
        anim.save(self.fname, writer=writergif)


if __name__ == "__main__":

    hist_path = "../OUTPUT/N3_opt_1kgs_50kW/history_50_BPR300.out"
    out_path = "../../../postprocessing/N3_opt_50/"
    hist = History(hist_path)
    dvs = hist.getDVInfo()
    scales = [10, 10, 6]
    w_key = "TOC.hx.dv.channel_width_cold"
    h_key = "TOC.hx.dv.channel_height_cold"
    l_key = "TOC.hx.dv.fin_length_cold"
    width = scales[0] * hist.getValues(w_key, major=True)[w_key].flatten()
    height = scales[1] * hist.getValues(h_key, major=True)[h_key].flatten()
    length = scales[2] * hist.getValues(l_key, major=True)[l_key].flatten()
    # print(width)
    # print(height)
    # print(length)

    # Plot = PlotGeo(height, width, length, rows=31, dir=out_path, fname="animation")
    # Plot.plot_animation()
    # m = int(width.size / 10)
    # plot_HX(h_c=height[0], w_c=width[0], l_c=length[0], dir=out_path, fname="n3_opt_design1")
    # plot_HX(h_c=height[m], w_c=width[m], l_c=length[m], dir=out_path, fname="n3_opt_design2")
    # plot_HX(h_c=height[-1], w_c=width[-1], l_c=length[-1], dir=out_path, fname="n3_opt_design3")

    plot_HX(h_c=4.02018178, w_c=0.89216891, l_c=2.31185624, dir="../../../postprocessing/HX_opt_50/", fname="hx_opt")
