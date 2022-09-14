#!/usr/bin/env python
"""
@File    :   plot_hx_sweeps.py
@Time    :   2022/04/25
@Desc    :   None
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt


def plot_hx_params(data, xlabel, fname):
    deltaP = data[:, 0]
    T_in_hot = data[:, 1]
    T_out_hot = data[:, 2]
    # dh = data[:, 3]
    p = data[:, 4]

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 5))
    axs[0].plot(p, deltaP)
    axs[1].plot(p, T_in_hot)
    axs[2].plot(p, T_out_hot)
    # axs[3].plot(p, dh)
    axs[0].set_ylabel(r"$\Delta$ P, kPa")
    axs[1].set_ylabel(r"$T_{in,hot}$, K")
    axs[2].set_ylabel(r"$T_{out,hot}$, K")
    # axs[3].set_ylabel(r"$dh_{cold}$, K")
    axs[2].set_xlabel(xlabel)

    out_path = "../../../postprocessing/hx_sweeps/"
    plt.tight_layout()
    fig.savefig(out_path + fname + ".pdf")
    # plt.show()


def plot_hx_geo(h, w, lf, xlabel, fname):
    deltaP = h[:, 0]
    T_in_hot = h[:, 1]
    T_out_hot = h[:, 2]
    dh = h[:, 3]
    p = h[:, 4]

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 5))
    axs[0].plot(p, deltaP)
    axs[1].plot(p, T_in_hot)
    axs[2].plot(p, T_out_hot)
    axs[3].plot(p, dh)
    axs[0].set_ylabel(r"$\Delta$ P, psi")
    axs[1].set_ylabel(r"$T_{in,hot}$, K")
    axs[2].set_ylabel(r"$T_{out,hot}$, K")
    axs[3].set_ylabel(r"$dh_{cold}$, K")
    axs[3].set_xlabel(xlabel)

    out_path = "../../../postprocessing/hx_sweeps/"
    plt.tight_layout()
    fig.savefig(out_path + fname + ".pdf")
    # plt.show()


# def plot_hx_geometry(hdata, wdata, fdata):
#     deltaP = data[:, 0]
#     T_in_hot = data[:, 1]
#     T_out_hot = data[:, 2]
#     h = data[:, 3]

#     fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 10))
#     axs[0].plot(h, deltaP)
#     axs[1].plot(h, T_in_hot)
#     axs[2].plot(h, T_out_hot)
#     axs[0].set_ylabel(r"$\Delta$ P, psi")
#     axs[1].set_ylabel(r"$T_{in,hot}$, K")
#     axs[2].set_ylabel(r"$T_{out,hot}$, K")
#     axs[2].set_xlabel(xlabel)

#     out_path = "../../../postprocessing/hx_sweeps/"
#     fig.savefig(out_path + fname + ".pdf")


if __name__ == "__main__":
    input_dir = "../OUTPUT/hx_sweeps"

    # plot_hx_params(np.load(input_dir + "/h_sweeps.npy"), xlabel="Channel Height, mm", fname="channel_height_hx")
    # plot_hx_params(np.load(input_dir + "/w_sweeps.npy"), xlabel="Channel Width, mm", fname="channel_width_hx")
    # plot_hx_params(np.load(input_dir + "/l_sweeps.npy"), xlabel="Fin Length, mm", fname="fin_length_hx")
    # plot_hx_params(np.load(input_dir + "/mc_sweeps.npy"), xlabel="mdot_cold, lbm/s", fname="mdot_cold_hx")
    plot_hx_params(np.load(input_dir + "/mh_sweeps.npy"), xlabel=r"$\dot{m}_{hot}$, kg/s", fname="mdot_hot_hx")
    # plot_hx_params(np.load(input_dir + "/elec_sweeps.npy"), xlabel="Electric Heat Load, kW", fname="elec_hx")
