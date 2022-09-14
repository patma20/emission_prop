#!/usr/bin/env python
"""
@File    :   plot_sweeps.py
@Time    :   2022/04/23
@Desc    :   None
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
# import os

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# External Python modules
# ==============================================================================


def plot_parameters(data, xlabel, fname):
    dPqP = data[:, 0]
    Fnet = data[:, 1]
    T_in_HX = data[:, 2]
    # FAR = data[:, 3]
    TSFC = data[:, 4]
    p = data[:, 5]

    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(8, 5))
    axs[0].plot(p, dPqP)
    axs[1].plot(p, Fnet)
    axs[2].plot(p, T_in_HX)
    axs[3].plot(p, TSFC)
    # axs[4].plot(p, FAR)
    axs[0].set_ylabel("dPqP")
    axs[0].ticklabel_format(useOffset=False)
    axs[0].set_ylim([np.min(dPqP) - 0.01, np.max(dPqP) + 0.01])
    axs[1].set_ylabel("Thrust Net, lbf")
    axs[2].set_ylabel(r"$T_{in,hot}$, K")
    axs[3].set_ylabel("TSFC")
    # axs[4].set_ylabel("FAR")
    axs[3].set_xlabel(xlabel)

    out_path = "../../../postprocessing/parameter_sweeps/"
    fig.savefig(out_path + fname + ".pdf")
    # plt.show()


def plot_tsfc(h, w, lf, sbpr, hpc, lpc, fan, t4, elec, fname):
    fig, axs = plt.subplots(9, 1, sharey=True, figsize=(8, 12))
    axs[0].plot(h[:, 5], h[:, 4])
    axs[1].plot(w[:, 5], w[:, 4])
    axs[2].plot(lf[:, 5], lf[:, 4])
    axs[3].plot(sbpr[:, 5], sbpr[:, 4])
    axs[4].plot(fan[:, 5], fan[:, 4])
    axs[5].plot(lpc[:, 5], lpc[:, 4])
    axs[6].plot(hpc[:, 5], hpc[:, 4])
    axs[7].plot(t4[:, 5], t4[:, 4])
    axs[8].plot(elec[:, 5], elec[:, 4])
    axs[8].set_ylim([0.5, 0.6])

    # axs[4].plot(p, FAR)
    axs[0].set_xlabel("h_c, mm")
    axs[1].set_xlabel("w_c, mm")
    axs[2].set_xlabel("l_f, mm")
    axs[3].set_xlabel("BPR_{HX,duct}")
    axs[4].set_xlabel("fanPR")
    axs[5].set_xlabel("lpcPR")
    axs[6].set_xlabel("hpcPR")
    axs[7].set_xlabel("T_4")
    axs[8].set_xlabel("Heat Load, kW")
    axs[3].set_ylabel("TSFC")

    out_path = "../../../postprocessing/parameter_sweeps/"
    fig.savefig(out_path + fname + ".pdf")
    # plt.show()


if __name__ == "__main__":
    output_dir = "../OUTPUT/sweeps"

    # plot_parameters(np.load(output_dir + "/h_sweeps.npy"), xlabel="Channel Height, mm", fname="channel_height")
    # plot_parameters(np.load(output_dir + "/w_sweeps.npy"), xlabel="Channel Width, mm", fname="channel_width")
    # plot_parameters(np.load(output_dir + "/l_sweeps.npy"), xlabel="Fin Length, mm", fname="fin_length")
    plot_parameters(np.load(output_dir + "/sbpr_sweeps.npy"), xlabel="Splitter BPR", fname="sbpr")
    # plot_parameters(np.load(output_dir + "/bpr_sweeps.npy"), xlabel="BPR", fname="bpr")
    # plot_parameters(np.load(output_dir + "/hpc_sweeps.npy"), xlabel="High Pressure Compressor PR", fname="hpcPR")
    plot_parameters(np.load(output_dir + "/lpc_sweeps.npy"), xlabel="Low Pressure Compressor PR", fname="lpcPR")
    # plot_parameters(np.load(output_dir + "/fan_sweeps.npy"), xlabel="Fan Pressure Compressor PR", fname="fanPR")
    # plot_parameters(np.load(output_dir + "/t4_sweeps.npy"), xlabel="T4, K", fname="t4")
    plot_parameters(np.load(output_dir + "/elec_sweeps.npy"), xlabel="Electric Heat Load, kW", fname="elec")
    # plot_tsfc(
    #     h=np.load(output_dir + "/h_sweeps.npy"),
    #     w=np.load(output_dir + "/w_sweeps.npy"),
    #     lf=np.load(output_dir + "/l_sweeps.npy"),
    #     sbpr=np.load(output_dir + "/sbpr_sweeps.npy"),
    #     fan=np.load(output_dir + "/fan_sweeps.npy"),
    #     lpc=np.load(output_dir + "/lpc_sweeps.npy"),
    #     hpc=np.load(output_dir + "/hpc_sweeps.npy"),
    #     t4=np.load(output_dir + "/t4_sweeps.npy"),
    #     elec=np.load(output_dir + "/elec_sweeps.npy"),
    #     fname="tsfc_sweeps",
    # )
