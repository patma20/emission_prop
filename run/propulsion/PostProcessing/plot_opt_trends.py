#!/usr/bin/env python
"""
@File    :   plot_opt_trends.py
@Time    :   2022/04/29
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
from pyoptsparse import History

# ==============================================================================
# Extension modules
# ==============================================================================


def plot_data(h, w, lf, pdrop, tsfc, elecload, fname):
    fig, axs = plt.subplots(5, 1, sharex=True, figsize=(8, 8))

    axs[0].plot(elecload, h)
    axs[1].plot(elecload, w)
    axs[2].plot(elecload, lf)
    axs[3].plot(elecload, pdrop)
    axs[4].plot(elecload, tsfc)

    axs[0].set_ylabel(r"$h_c$, mm")
    axs[1].set_ylabel(r"$w_c$, mm")
    axs[2].set_ylabel(r"$l_f$, mm")
    axs[3].set_ylabel(r"$dPqP_{HX,duct}$")
    axs[4].set_ylabel(r"$TSFC$")
    axs[4].ticklabel_format(useOffset=False)
    axs[4].set_xlabel("Heat Load, kW")
    out_path = "../../../postprocessing/parameter_sweeps/"
    fig.savefig(out_path + fname + ".pdf")
    # plt.show()


def plot_ratio(h, w, elecload, fname):
    fig = plt.figure(figsize=(8, 4))

    plt.plot(elecload, h / w)
    plt.xlabel("Heat Load, kW")
    plt.ylabel("$h_c/w_c$")

    out_path = "../../../postprocessing/parameter_sweeps/"
    fig.savefig(out_path + fname + ".pdf")


def get_data(hist_path):
    hist = History(hist_path)
    scales = [10, 10, 6, 1, 1]
    w_key = "TOC.hx.dv.channel_width_cold"
    h_key = "TOC.hx.dv.channel_height_cold"
    l_key = "TOC.hx.dv.fin_length_cold"
    pdrop_key = "TOC.HXduct.dPqP"
    tsfc_key = "TOC.perf.TSFC"
    width = scales[0] * hist.getValues(w_key, major=True)[w_key].flatten()
    height = scales[1] * hist.getValues(h_key, major=True)[h_key].flatten()
    length = scales[2] * hist.getValues(l_key, major=True)[l_key].flatten()
    pdrop = hist.getValues(pdrop_key, major=True)[pdrop_key].flatten()
    TSFC = hist.getValues(tsfc_key, major=True)[tsfc_key].flatten()

    return width[-1], height[-1], length[-1], pdrop[-1], TSFC[-1]


if __name__ == "__main__":
    output_dir = "../OUTPUT/sweeps"
    elecload = np.array([5, 10, 20, 32, 37, 40, 42, 45, 50, 55, 60])
    h = np.zeros(elecload.size)
    w = np.zeros(elecload.size)
    lf = np.zeros(elecload.size)
    pdrop = np.zeros(elecload.size)
    tsfc = np.zeros(elecload.size)

    for i, pt in enumerate(elecload):
        hist_path = f"../OUTPUT/N3_opt_1kgs_{pt}kW/history_{pt}_BPR300.out"
        w[i], h[i], lf[i], pdrop[i], tsfc[i] = get_data(hist_path=hist_path)

    # plot_data(h, w, lf, pdrop, tsfc, elecload, "TSFC_opt_sweeps")
    plot_ratio(h, w, elecload, "hc_opt_sweeps")

    # out_path = f"../../../postprocessing/N3opt_20kW/"
