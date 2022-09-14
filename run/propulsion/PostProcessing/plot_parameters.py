#!/usr/bin/env python
"""
@File    :   plot_parameters.py
@Time    :   2022/04/20
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
from pyoptsparse import History

# ==============================================================================
# Extension modules
# ==============================================================================


def plot_params(params, dir, fname, subplot=True):
    n = len(params)
    if subplot is True:
        fig, axs = plt.subplots(n, 1, sharex=True, figsize=(10, 30))
        for i in range(n):
            data = list(params.values())[i]
            axs[i].semilogx(data)
            axs[i].set_ylim([min(data), max(data)])
            axs[i].set_ylabel(list(params.keys())[i])

    else:
        fig = plt.figure(figsize=(10, 10))
        for i in range(n):
            data = list(params.values())[i]
            plt.semilogy(data / data[-1])
        plt.legend(list(params.keys()))
        print(list(params.keys()))
    fig.tight_layout()
    fig.savefig(os.path.join(dir, fname + ".pdf"))
    # plt.show()


if __name__ == "__main__":
    hist_path = "../OUTPUT/N3_opt_20.0kW_1tall/history_20.0.out"
    out_path = "../../../postprocessing/N3opt_20kW/"
    hist = History(hist_path)
    # dvs = hist.getDVInfo()
    # cons = hist.getConInfo()
    # print(dvs)
    # print(cons)
    scales = [10, 10, 6, 1, 3600, 1, 1, 300, 6000]
    w_key = "TOC.hx.dv.channel_width_cold"
    h_key = "TOC.hx.dv.channel_height_cold"
    l_key = "TOC.hx.dv.fin_length_cold"
    fan_key = "fan:PRdes"
    T4_key = "TOC.balance.rhs:FAR"
    area_key = "TOC.HX_area_con.area_con"
    dPqP_key = "TOC.HXduct.dPqP"
    temp_key = "TOC.heatcomp.T_out"
    Fn_key = "TOC.perf.Fn"

    width = scales[0] * hist.getValues(w_key, major=True)[w_key].flatten()
    height = scales[1] * hist.getValues(h_key, major=True)[h_key].flatten()
    length = scales[2] * hist.getValues(l_key, major=True)[l_key].flatten()
    fanPR = scales[3] * hist.getValues(fan_key, major=True)[fan_key].flatten()
    Temp4 = scales[4] * hist.getValues(T4_key, major=True)[T4_key].flatten()
    area_con = scales[5] * hist.getValues(area_key, major=True)[area_key].flatten()
    dPqP = scales[6] * hist.getValues(dPqP_key, major=True)[dPqP_key].flatten()
    coolantTemp = scales[7] * hist.getValues(temp_key, major=True)[temp_key].flatten()
    fnet = scales[8] * hist.getValues(Fn_key, major=True)[Fn_key].flatten()

    params = {
        "channel width": width,
        "channel height": height,
        "fin length": length,
        "Fan PR": fanPR,
        "Combustor T4": Temp4,
        "Area Constraint": area_con,
        "HX pressure drop": dPqP,
        "Coolant Temperature": coolantTemp,
        "Net Force": fnet,
    }

    fname = "dv_conparison_1tall"
    plot_params(params, out_path, fname, subplot=True)
