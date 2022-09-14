#!/usr/bin/env python
"""
@File    :   plot_convergence.py
@Time    :   2022/04/12
@Desc    :   None
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import matplotlib.pyplot as plt

# import pickle
import numpy as np
import openmdao.api as om

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================


def n3_convergence():
    """
    For plotting convergence rates
    """

    fname = "n3HX_20kW_BPR300"
    path = "/Users/peteratma/research/thermo_prop/postprocessing/output/" + fname + ".sql"
    cr = om.CaseReader(path)
    driver_cases = cr.list_cases("driver")

    n = len(driver_cases)
    objective = np.zeros(n)
    design_var = np.zeros((n, 6))
    cons = np.zeros((n, 4))
    for i in range(n):
        case = cr.get_case(driver_cases[i])
        objective[i] = case["TOC.perf.TSFC"]
        design_var[i, 0] = case["TOC.hx.channel_width_cold"]
        design_var[i, 1] = case["TOC.hx.channel_height_cold"]
        design_var[i, 2] = case["fan:PRdes"]
        design_var[i, 3] = case["lpc:PRdes"]
        design_var[i, 4] = case["TOC.balance.rhs:hpc_PR"]
        design_var[i, 5] = case["TOC.balance.rhs:FAR"]
        cons[i, 0] = case["TOC.HXduct.dPqP"]
        cons[i, 1] = case["TOC.HX_area_con.area_con"]
        cons[i, 2] = case["TOC.heatcomp.T_out"]
        cons[i, 3] = case["TOC.perf.Fn"]
    itr = np.arange(0, n)

    plt.figure(figsize=(10, 5))
    plt.semilogy(itr[1:], np.abs(objective[:-1] - objective[1:]), label="TSFC")
    # plt.semilogy(itr[1:], np.abs(con1[:-1] - con1[1:]), label="hoop_cmp")
    # plt.semilogy(itr[1:], np.abs(con2[:-1] - con2[1:]), label="axial_cmp")
    # plt.semilogy(itr[1:], np.abs(con3[:-1] - con3[1:]), label="thickness_cmp")
    # plt.semilogy(itr[1:], np.abs(con4[:-1] - con4[1:]), label="deltav_cmp")
    plt.ylabel("Relative Change", fontsize=14)
    plt.xlabel("Iteration", fontsize=14)
    plt.legend()
    plt.savefig("/Users/peteratma/research/thermo_prop/postprocessing/" + fname + ".pdf")


def hx_convergence(dir_path, output_path, fname):
    """
    For plotting convergence rates
    """

    path = dir_path + fname + ".sql"
    cr = om.CaseReader(path)
    driver_cases = cr.list_cases("driver")

    n = len(driver_cases)
    dp = np.zeros(n)
    h = np.zeros(n)
    w = np.zeros(n)
    lf = np.zeros(n)
    dhc = np.zeros(n)
    f = np.zeros(n)

    for i in range(n):
        case = cr.get_case(driver_cases[i])
        dp[i] = case["hx.delta_p_cold"]
        h[i] = case["hx.channel_height_cold"]
        w[i] = case["hx.channel_width_cold"]
        lf[i] = case["hx.fin_length_cold"]
        dhc[i] = case["hx.dh_cold"]
        f[i] = case["hx.f_cold"]

    itr = np.arange(0, n)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axs[0].plot(itr, h / max(h), label="channel height")
    axs[0].plot(itr, w / max(w), label="channel width")
    axs[0].plot(itr, lf / max(lf), label="fin length")
    axs[0].plot(itr, dhc / max(dhc), label="hydraulic diameter")
    axs[0].plot(itr, f / max(f), label="friction factor")
    axs[0].set_ylabel("Normalized Cold Size Parameters")
    axs[0].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", mode="expand", borderaxespad=0, ncol=5)
    # axs[0].legend(bbox_to_anchor=(1.25, 1), loc="upper right")
    # axs[0].legend(loc="best")
    axs[1].semilogy(itr, dp / max(dp))
    axs[1].set_ylabel(r"$\Delta P, kPa$")
    # plt.show()

    # fig, axs = plt.subplots(5, 1, sharex=True, figsize=(8, 10))
    # axs[0].plot(itr, h / max(h))
    # axs[1].plot(itr, w / max(w))
    # axs[2].plot(itr, lf / max(lf))
    # axs[3].plot(itr, dhc / max(dhc))
    # axs[4].plot(itr, dp / max(dp))

    # axs[0].set_ylabel("channel height")
    # axs[1].set_ylabel("channel width")
    # axs[2].set_ylabel("fin length")
    # axs[3].set_ylabel("hydraulic diameter cold")
    # axs[4].set_ylabel("Delta P")

    # axs[4].set_xlabel("Iteration", fontsize=14)

    fig.savefig(output_path + fname + ".pdf")


if __name__ == "__main__":
    hx_convergence(
        dir_path="../OUTPUT/HX_opt_50kW", output_path="../../../postprocessing/hx_sweeps/", fname="n3_opt_50"
    )
