#!/usr/bin/env python
"""
@File    :   plot_sweeps.py
@Time    :   2022/04/23
@Desc    :   None
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os

# ==============================================================================
# External Python modules
# ==============================================================================
import matplotlib.pyplot as plt
import niceplots
import pickle as pkl
import numpy as np

# ==============================================================================
# External Python modules
# ==============================================================================


def plot_sweeps(dir_JetA, dir_H2, var, output_dir):

    with open(dir_JetA + "/N3_" + var + ".pkl", "rb") as f:
        jeta_data = pkl.load(f)

    var_jeta = jeta_data[0, :]
    wdot_jeta = jeta_data[1, :]
    TSEC_jeta = jeta_data[4, :]

    with open(dir_H2 + "/N3_" + var + ".pkl", "rb") as f:
        h2_data = pkl.load(f)

    var_h2 = h2_data[0, :]
    wdot_h2 = h2_data[1, :]
    TSEC_h2 = h2_data[4, :]

    fig, axs = plt.subplots(1, 1, sharey=True, figsize=(10, 6))
    axs.plot(var_jeta, TSEC_jeta, linestyle="-", label="Jet-A")
    axs.plot(var_h2, TSEC_h2, linestyle="-", label="H2")
    axs.set_xlabel(var)
    # axs.set_ylim([9000, 7000])
    axs.set_ylabel("TSEC")
    plt.legend()
    fig.savefig(output_dir + var + "_TSEC.pdf")
    fig.savefig(output_dir + var + "_TSEC.png")
    # plt.show()

    fig, axs = plt.subplots(1, 1, sharey=True, figsize=(10, 6))
    axs.plot(var_jeta, wdot_jeta, linestyle="-", label="Jet-A")
    axs.plot(var_h2, wdot_h2, linestyle="-", label="H2")
    # axs.set_ylim([0.0, 0.1])
    axs.set_xlabel(var)
    axs.set_ylabel("Water Extracted (lbm/s)")
    plt.legend()
    fig.savefig(output_dir + var + "_wdot.pdf")
    fig.savefig(output_dir + var + "_wdot.png")
    # plt.show()


if __name__ == "__main__":
    niceplots.setRCParams()
    niceColors = niceplots.get_niceColors()
    plt.rcParams["font.size"] = 20
    output_dir = "plots/sweeps/"

    plot_sweeps(
        "../OUTPUT/N3_trends/bound_sweeps/JetA",
        "../OUTPUT/N3_trends/bound_sweeps/H2",
        "TOC.balance.rhs:hpc_PR",
        output_dir,
    )
    plot_sweeps("../OUTPUT/N3_trends/bound_sweeps/JetA", "../OUTPUT/N3_trends/bound_sweeps/H2", "fan:PRdes", output_dir)
    plot_sweeps("../OUTPUT/N3_trends/bound_sweeps/JetA", "../OUTPUT/N3_trends/bound_sweeps/H2", "lpc:PRdes", output_dir)
    plot_sweeps(
        "../OUTPUT/N3_trends/bound_sweeps/JetA", "../OUTPUT/N3_trends/bound_sweeps/H2", "T4_ratio.TR", output_dir
    )
    plot_sweeps("../OUTPUT/N3_trends/bound_sweeps/JetA", "../OUTPUT/N3_trends/bound_sweeps/H2", "RTO_T4", output_dir)
    plot_sweeps(
        "../OUTPUT/N3_trends/bound_sweeps/JetA",
        "../OUTPUT/N3_trends/bound_sweeps/H2",
        "TOC.extract.sub_flow.w_frac",
        output_dir,
    )
