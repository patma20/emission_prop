#!/usr/bin/env python
"""
@File    :   N3_HXnozz_Opt.py
@Time    :   2022/04/05
@Desc    :   None
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import numpy as np
import time
import os
import pickle as pkl

# ==============================================================================
# External Python modules
# ==============================================================================
import openmdao.api as om
import pycycle.api as pyc

# ==============================================================================
# Extension modules
# ==============================================================================
from N3_CLVR_V3 import N3, viewer, MPN3
from N3_CLVR_OPT import N3_MDP_Opt_model

if __name__ == "__main__":
    save_res = True
    use_h2 = True
    save_init_data = True

    if use_h2:
        fuel = "H2"
    else:
        fuel = "JetA"

    output_dir = f"../OUTPUT/N3_opt/CLVR/analysis/N3_{fuel}_thermo_wTOC-CRZ-RTO-SLS"

    # Create optimization problem
    prob = N3_MDP_Opt_model(output_dir, save_res, use_h2)
    prob.setup()

    # Define the design point
    prob.set_val("TOC.fc.W", 820.44097898, units="lbm/s")
    prob.set_val("TOC.splitter.BPR", 23.94514401)

    # Set specific cycle parameters
    prob.set_val("SLS.fc.MN", 0.001)
    prob.set_val("SLS.balance.rhs:FAR", 28620.84, units="lbf")
    prob.set_val("CRZ.balance.rhs:FAR", 5510.72833567, units="lbf")
    prob.set_val("bal.rhs:TOC_BPR", 1.40)
    prob.set_val("RTO.hpt_cooling.x_factor", 0.9)

    if use_h2:
        # Define the design point
        prob.set_val("TOC.balance.rhs:hpc_PR", 51.0)

        # Set specific cycle parameters
        prob.set_val("T4_ratio.TR", 0.91)
        prob.set_val("RTO_T4", 3400.0, units="degR")
        prob.set_val("fan:PRdes", 1.300)
        prob.set_val("lpc:PRdes", 4.000)
        prob.set_val("TOC.inject.area", 117.730, units="inch**2")
        prob.set_val("TOC.extract.area", 1053.492, units="inch**2")

        # Set inital guesses for balances
        prob["TOC.balance.FAR"] = 0.0102
        prob["TOC.balance.lpt_PR"] = 9.8
        prob["TOC.balance.hpt_PR"] = 4.2
        prob["bal.TOC_W"] = 820.95
        prob["TOC.fc.balance.Pt"] = 5.2
        prob["TOC.fc.balance.Tt"] = 444.3
        prob["TOC.inject.mix:W"] = 0.0
        prob["TOC.extract.sub_flow.w_frac"] = 0.0
        prob["CRZ.extract.sub_flow.w_frac"] = 0.0
        prob["RTO.extract.sub_flow.w_frac"] = 0.0
        prob["SLS.extract.sub_flow.w_frac"] = 0.0

        FAR_guess = [0.0106, 0.02541, 0.0094]
        W_guess = [1906.3, 1900.4, 797.5]
        BPR_guess = [26.1, 22.3467, 24.8]
        fan_Nmech_guess = [2119.9, 1953.1, 2095.7]
        lp_Nmech_guess = [6572.0, 6054.5, 6496.8]
        hp_Nmech_guess = [22150.0, 21594.0, 20431.0]
        hpt_PR_guess = [4.2, 4.245, 4.2]
        lpt_PR_guess = [8.0, 7.001, 9.9]
        Pt_guess = [15.349, 14.696, 5.272]
        Tt_guess = [552.49, 545.67, 444.41]
        fan_Rline_guess = [1.7500, 1.7500, 1.9397]
        lpc_Rline_guess = [1.9, 1.8632, 2.0]
        hpc_Rline_guess = [2.0, 2.0281, 1.9]
        trq_guess = [52047.8, 41779.4, 21780.5]
        w_inject = [0.0000, 0.0000, 0.0000]

    else:
        # Define the design point
        prob.set_val("TOC.balance.rhs:hpc_PR", 53.6332)

        # Set specific cycle parameters
        prob.set_val("T4_ratio.TR", 0.926470588)
        prob.set_val("RTO_T4", 3400.0, units="degR")
        prob.set_val("fan:PRdes", 1.300)
        # prob.set_val("fan:PRdes", 1.300)
        prob.set_val("lpc:PRdes", 4.000)
        # prob.set_val("lpc:PRdes", 3.000)
        prob.set_val("TOC.inject.area", 117.730, units="inch**2")
        prob.set_val("TOC.extract.area", 1053.492, units="inch**2")

        # Set inital guesses for balances
        prob["TOC.balance.FAR"] = 0.02650
        prob["TOC.balance.lpt_PR"] = 10.937
        prob["TOC.balance.hpt_PR"] = 4.185
        prob["bal.TOC_W"] = 820.95
        prob["TOC.fc.balance.Pt"] = 5.272
        prob["TOC.fc.balance.Tt"] = 444.41
        prob["TOC.inject.mix:W"] = 0.0
        prob["TOC.extract.sub_flow.w_frac"] = 0.0
        prob["CRZ.extract.sub_flow.w_frac"] = 0.0
        prob["RTO.extract.sub_flow.w_frac"] = 0.0
        prob["SLS.extract.sub_flow.w_frac"] = 0.0

        FAR_guess = [0.02832, 0.02541, 0.02510]
        W_guess = [1916.13, 1900.0, 802.79]
        BPR_guess = [25.5620, 22.3467, 24.3233]
        fan_Nmech_guess = [2132.6, 1953.1, 2118.7]
        lp_Nmech_guess = [6611.2, 6054.5, 6567.9]
        hp_Nmech_guess = [22288.2, 21594.0, 20574.1]
        Pt_guess = [15.349, 14.696, 5.272]
        Tt_guess = [552.49, 545.67, 444.41]
        hpt_PR_guess = [4.210, 4.245, 4.197]
        lpt_PR_guess = [8.161, 7.001, 10.803]
        fan_Rline_guess = [1.7500, 1.7500, 1.9397]
        lpc_Rline_guess = [2.0052, 1.8632, 2.1075]
        hpc_Rline_guess = [2.0589, 2.0281, 1.9746]
        trq_guess = [52509.1, 41779.4, 22369.7]
        w_inject = [0.0000, 0.0000, 0.01]

    for i, pt in enumerate(prob.model.od_pts):

        # initial guesses
        prob[pt + ".balance.FAR"] = FAR_guess[i]
        prob[pt + ".balance.W"] = W_guess[i]
        prob[pt + ".balance.BPR"] = BPR_guess[i]
        prob[pt + ".balance.fan_Nmech"] = fan_Nmech_guess[i]
        prob[pt + ".balance.lp_Nmech"] = lp_Nmech_guess[i]
        prob[pt + ".balance.hp_Nmech"] = hp_Nmech_guess[i]
        prob[pt + ".fc.balance.Pt"] = Pt_guess[i]
        prob[pt + ".fc.balance.Tt"] = Tt_guess[i]
        prob[pt + ".hpt.PR"] = hpt_PR_guess[i]
        prob[pt + ".lpt.PR"] = lpt_PR_guess[i]
        prob[pt + ".fan.map.RlineMap"] = fan_Rline_guess[i]
        prob[pt + ".lpc.map.RlineMap"] = lpc_Rline_guess[i]
        prob[pt + ".hpc.map.RlineMap"] = hpc_Rline_guess[i]
        prob[pt + ".gearbox.trq_base"] = trq_guess[i]
        prob[pt + ".inject.mix:W"] = w_inject[i]
        # prob[pt + ".extract.sub_flow.w_frac"] = w_extract[i]

    st = time.time()

    # N2 generation
    # om.n2(prob)

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    prob.run_driver()

    if save_res is True:
        with open(f"{output_dir}/outputs.out", "w") as file:
            prob.model.list_outputs(prom_name=True, units=True, out_stream=file)

        with open(f"{output_dir}/inputs.out", "w") as file:
            prob.model.list_inputs(prom_name=True, units=True, out_stream=file)

        with open(f"{output_dir}/output.txt", "w") as file:
            for pt in ["TOC"] + prob.model.od_pts:
                viewer(prob, pt, file)

            print(file=file, flush=True)
            print("Run time", time.time() - st, file=file, flush=True)

    print("time", time.time() - st)

