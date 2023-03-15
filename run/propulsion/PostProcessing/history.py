#!/usr/bin/env python
"""
@File    :   history.py
@Time    :   2022/04/09
@Desc    :   None
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os

# ==============================================================================
# External Python modules
# ==============================================================================
from pyoptsparse import History
import matplotlib.pyplot as plt
import niceplots as nplt
import matplotlib.patheffects as patheffects
from matplotlib.ticker import MaxNLocator
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================

plt.style.use("styles/historyStyle")
NICE_COLORS = nplt.get_niceColors()


def remove_ticks(axes):
    for i, ax in enumerate(axes):
        # adjust_spines(ax, outward=True)
        if i < len(axes) - 1:
            ax.xaxis.set_ticks([])


class HistoryPlotter:
    def __init__(self, fp: str, output_dir: str = None):
        self.hist = History(fp)
        self.output_dir = output_dir

    def plot_opt_summary(
        self,
        opt_tol=1e-6,
        feas_tol=1e-6,
        colors=[NICE_COLORS["Red"], NICE_COLORS["Blue"], NICE_COLORS["Green"]],
        **kwargs,
    ):
        # Get optimality, feasibility, and time from the hist file
        optimality = self.hist.getValues("optimality")
        feasibility = self.hist.getValues("feasibility")
        # time = self.hist.getValues("time")

        # Compute the time per iteration instead of cumulative
        # time["time"] = time["time"][1:] - time["time"][:-1]

        # Merge the data dicts
        opt_vars = optimality | feasibility  # | time

        # Dict for opt and feas tols
        tolerance = {"optimality": opt_tol, "feasibility": feas_tol}

        # Create the figure and subplots
        fig, axes = plt.subplots(2, 1, sharex=False, figsize=(12, 8))

        # Looop over all the objects we need for plotting
        for ax, key, color in zip(axes, opt_vars.keys(), colors):
            if key in ["feasibility", "optimality"]:
                # Log-y plot of feas and opt
                ax.semilogy(np.arange(0, len(opt_vars[key]), 1), opt_vars[key], label=key, color=color, **kwargs)

                # Add the tolerance line and annotate
                ax.axhline(y=tolerance[key], color=NICE_COLORS["Grey"], linestyle="--")
                ax.annotate(
                    f"{tolerance[key]:.1e}",
                    xy=(2, tolerance[key]),
                    xytext=(2, 1.5 * tolerance[key]),
                    color=NICE_COLORS["Grey"],
                )
            # else:
            # Plot the time
            # ax.plot(np.arange(0, len(opt_vars[key]), 1), opt_vars[key], label=key, color=color, **kwargs)

            # Set the x-axis ticks to integers
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend()

        # Remove tick marks from the subplots except the last one
        remove_ticks(axes)

        # Set the x-label on the last subplot
        axes[-1].set_xlabel("Iterations")

        # If an output dir is given, save the plot.
        # Else, return the fig and axes
        if self.output_dir is None:
            return fig, axes
        else:
            fig.savefig(os.path.join(self.output_dir, "opt_summary.pdf"))

    def plot_dvs(self, major=True, **kwargs):
        # Get the dv info from the hist file
        dvs = self.hist.getDVInfo()

        # Count the number of dvs for making the subplots
        num_dvs = len([key for key in dvs.keys()])

        # Loop over the keys and store the data and line labels
        for key in dvs.keys():
            dvs[key]["data"] = self.hist.getValues(key, major=major)[key]
            dvs[key]["label"] = key.split(".")[-1]

        # Define a color list using nice colors from niceplots
        # NOTE: This might throw an error if num_dvs > num_colors
        colors = [color for color in NICE_COLORS.values()][:num_dvs]

        # Setup the subplots
        fig, axes = plt.subplots(num_dvs, 1, sharex=False, figsize=(12, 18))

        for ax, key, color in zip(axes, dvs.keys(), colors):
            lower_bound = dvs[key]["lower"][0]
            upper_bound = dvs[key]["upper"][0]
            data = dvs[key]["data"]

            # Plot the data and upper/lower bounds
            line = ax.plot(np.arange(0, len(data), 1, dtype=int), data, label=dvs[key]["label"], color=color, **kwargs)
            ax.axhline(y=upper_bound, path_effects=[patheffects.withTickedStroke()], color=line[0].get_color())
            ax.axhline(
                y=lower_bound, path_effects=[patheffects.withTickedStroke(angle=-135)], color=line[0].get_color()
            )

            # Set x-axis tick labels to integers
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # If there's an upper an lower bound, we scale the y-axis
            # to fit the bounds nicely.
            if upper_bound is not None and lower_bound is not None:
                height = upper_bound - lower_bound
                limits = [lower_bound - 0.1 * height, upper_bound + 0.1 * height]
                ax.set_ylim(limits)

            ax.legend()

        # Remove ticks from all subplot x-axis except the last one
        remove_ticks(axes)

        # If an output dir is given, save the plot.
        # Else, return the fig and axes
        if self.output_dir is None:
            return fig, axes
        else:
            fig.savefig(os.path.join(self.output_dir, "dvs.pdf"))

    def plot_cons(self, major=True, **kwargs):
        # Get the constraint info from the hist file
        cons = self.hist.getConInfo()

        # Count the number of constraints
        num_cons = len([key for key in cons.keys()])

        # Set the constraint data and line labels
        for key in cons.keys():
            cons[key]["data"] = self.hist.getValues(key, major=major)[key]
            cons[key]["label"] = key.split(".")[-1]

        # Set colors for the constraints using nice colors
        # NOTE: This might fail if num_cons > num_colors
        colors = [color for color in NICE_COLORS.values()][:num_cons]

        # Setup the fig and subplots
        fig, axes = plt.subplots(num_cons, 1, sharex=False, figsize=(12, 18))

        for ax, key, color in zip(axes, cons.keys(), colors):
            lower_bound = cons[key]["lower"][0]
            upper_bound = cons[key]["upper"][0]
            data = cons[key]["data"]

            # Plot the constraint data
            line = ax.plot(
                np.arange(0, len(data), 1, dtype=int), data, label=cons[key]["label"], color=color, zorder=2, **kwargs
            )

            # Set the x-axis tick labels to integers
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            if upper_bound != lower_bound:
                if upper_bound is not None and lower_bound is not None:
                    height = upper_bound - lower_bound
                    limits = [lower_bound - 0.1 * height, upper_bound + 0.1 * height]
                    ax.set_ylim(limits)
                ax.axhline(
                    y=upper_bound, path_effects=[patheffects.withTickedStroke()], color=line[0].get_color(), zorder=1
                )
                ax.axhline(
                    y=lower_bound,
                    path_effects=[patheffects.withTickedStroke(angle=-135)],
                    color=line[0].get_color(),
                    zorder=1,
                )
            else:
                ax.axhline(y=upper_bound, color=NICE_COLORS["Grey"], zorder=1, linestyle="-.")

            ax.legend()

        # Remove ticks from all subplot x-axis except the last one
        remove_ticks(axes)

        # If an outptu dir is given, save the plot.
        # Else, return the fig and axes
        if self.output_dir is None:
            return fig, axes
        else:
            fig.savefig(os.path.join(self.output_dir, "cons.pdf"))

    def plot_obj(self, major=True, **kwargs):
        key = self.hist.getObjNames()[0]
        obj = self.hist.getObjInfo()
        obj[key]["data"] = self.hist.getValues(key, major=major)[key]
        obj[key]["label"] = key.split(".")[-1]

        color = NICE_COLORS["Blue"]

        fig, ax = plt.subplots()

        ax.plot(
            np.arange(0, len(obj[key]["data"]), 1), obj[key]["data"], label=obj[key]["label"], color=color, **kwargs
        )
        ax.legend()

        # If an outptu dir is given, save the plot.
        # Else, return the fig and axes
        if self.output_dir is None:
            return fig, ax
        else:
            fig.savefig(os.path.join(self.output_dir, "obj.pdf"))


if __name__ == "__main__":
    hist_path = "../OUTPUT/N3_opt/CLVR/TSFC/N3_wfrac_H2_CRZ/history_CLVR_H2.out"
    # hist_path = "../OUTPUT/N3_opt/CLVR/TSFC/N3_wfrac_JetA_CRZ/history_CLVR.out"
    out_path = "../../../postprocessing/CLVR_H2/"
    # out_path = "../../../postprocessing/CLVR_JetA/"
    if os.path.isdir(out_path) is False:
        os.mkdir(out_path)
    plotter = HistoryPlotter(hist_path, out_path)
    plotter.plot_opt_summary(marker=".", markersize=6.0)
    plotter.plot_dvs(marker=".", markersize=6.0)
    plotter.plot_cons(marker=".", markersize=6.0)
    plotter.plot_obj(marker=".", markersize=6.0)
