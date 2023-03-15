#!/usr/bin/env python
"""
@File    :   sweepsN3_CLVR.py
@Time    :   2023/01/31
@Desc    :   None
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
# import sys
# import os

# ==============================================================================
# External Python modules
# ==============================================================================
import openmdao.api as om

# import pycycle.api as pyc
import pickle as pkl
import numpy as np

from mpi4py import MPI

# ==============================================================================
# Extension modules
# ==============================================================================
from N3_vapor_rec import MPN3


def N3ref_model(use_h2=False, wet_air=True):

    prob = om.Problem(comm=MPI.COMM_SELF)

    prob.model = MPN3(use_h2=use_h2, wet_air=wet_air)

    return prob


if __name__ == "__main__":
    import time

    use_h2 = True
    prob = N3ref_model(use_h2=use_h2)

    prob.setup()

    # Define the design point
    prob.set_val("TOC.fc.W", 820.44097898, units="lbm/s")
    prob.set_val("TOC.splitter.BPR", 23.94514401),
    prob.set_val("TOC.balance.rhs:hpc_PR", 53.6332)  # for JetA

    # Set up the specific cycle parameters
    prob.set_val("fan:PRdes", 1.300),
    prob.set_val("lpc:PRdes", 3.000),
    prob.set_val("T4_ratio.TR", 0.926470588)
    prob.set_val("RTO_T4", 3400.0, units="degR")
    prob.set_val("SLS.balance.rhs:FAR", 28620.84, units="lbf")
    prob.set_val("CRZ.balance.rhs:FAR", 5510.72833567, units="lbf")
    prob.set_val("RTO.hpt_cooling.x_factor", 0.9)

    # Set initial guesses for balances
    if prob.model.options["use_h2"]:
        prob["TOC.balance.FAR"] = 0.0102
        prob["TOC.balance.lpt_PR"] = 9.8
        prob["TOC.balance.hpt_PR"] = 4.2
        prob["TOC.fc.balance.Pt"] = 5.2
        prob["TOC.fc.balance.Tt"] = 444.3
        prob["TOC.inject.mix:W"] = 0.0

        FAR_guess = [0.0106, 0.02541, 0.0094]
        W_guess = [1906.3, 1900.4, 797.5]
        BPR_guess = [26.1, 22.3467, 24.8]
        fan_Nmech_guess = [2119.9, 1953.1, 2095.7]
        lp_Nmech_guess = [6572.0, 6054.5, 6496.8]
        hp_Nmech_guess = [22150.0, 21594.0, 20431.0]
        hpt_PR_guess = [4.2, 4.245, 4.2]
        lpt_PR_guess = [8.0, 7.001, 9.9]
        fan_Rline_guess = [1.7500, 1.7500, 1.9397]
        lpc_Rline_guess = [1.9, 1.8632, 2.0]
        hpc_Rline_guess = [2.0, 2.0281, 1.9]
        trq_guess = [52047.8, 41779.4, 21780.5]

    else:
        prob["TOC.balance.FAR"] = 0.02650
        prob["TOC.balance.lpt_PR"] = 10.937
        prob["TOC.balance.hpt_PR"] = 4.185
        prob["TOC.fc.balance.Pt"] = 5.272
        prob["TOC.fc.balance.Tt"] = 444.41
        prob["TOC.inject.mix:W"] = 0.000

        FAR_guess = [0.02832, 0.02541, 0.02510]
        W_guess = [1916.13, 1900.0, 802.79]
        BPR_guess = [25.5620, 22.3467, 24.3233]
        fan_Nmech_guess = [2132.6, 1953.1, 2118.7]
        lp_Nmech_guess = [6611.2, 6054.5, 6567.9]
        hp_Nmech_guess = [22288.2, 21594.0, 20574.1]
        hpt_PR_guess = [4.210, 4.245, 4.197]
        lpt_PR_guess = [8.161, 7.001, 10.803]
        fan_Rline_guess = [1.7500, 1.7500, 1.9397]
        lpc_Rline_guess = [2.0052, 1.8632, 2.1075]
        hpc_Rline_guess = [2.0589, 2.0281, 1.9746]
        trq_guess = [52509.1, 41779.4, 22369.7]

    for i, pt in enumerate(prob.model.od_pts):

        # initial guesses
        prob[pt + ".balance.FAR"] = FAR_guess[i]
        prob[pt + ".balance.W"] = W_guess[i]
        prob[pt + ".balance.BPR"] = BPR_guess[i]
        prob[pt + ".balance.fan_Nmech"] = fan_Nmech_guess[i]
        prob[pt + ".balance.lp_Nmech"] = lp_Nmech_guess[i]
        prob[pt + ".balance.hp_Nmech"] = hp_Nmech_guess[i]
        prob[pt + ".hpt.PR"] = hpt_PR_guess[i]
        prob[pt + ".lpt.PR"] = lpt_PR_guess[i]
        prob[pt + ".fan.map.RlineMap"] = fan_Rline_guess[i]
        prob[pt + ".lpc.map.RlineMap"] = lpc_Rline_guess[i]
        prob[pt + ".hpc.map.RlineMap"] = hpc_Rline_guess[i]
        prob[pt + ".gearbox.trq_base"] = trq_guess[i]
        prob[pt + ".inject.mix:W"] = 0.000

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    n = 10
    # TOC_frac = np.linspace(0, 0.10, n)  # JetA TOC
    # CRZ_frac = np.linspace(0, 0.27, n)  # JetA CRZ
    TOC_frac = np.linspace(0, 0.15, n)  # H2 TOC
    CRZ_frac = np.linspace(0, 0.19, n)  # H2 CRZ
    istart = 5
    jstart = 4

    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size

    count = 0
    print(time.strftime("%H:%M:%S", time.localtime()))
    for i, TOCw in enumerate(TOC_frac):
        if i < istart:
            continue
        for j, CRZw in enumerate(CRZ_frac):
            if j < jstart:
                continue
            if count % size == rank:
                print(
                    10 * "#" + f" Running TOC_frac={TOCw:.2f}, CRZ_frac={CRZw:.2f} Iteration: {i+1},{j+1} " + 10 * "#"
                )

                prob["TOC.extract.sub_flow.w_frac"] = TOCw
                prob["CRZ.extract.sub_flow.w_frac"] = CRZw

                try:
                    prob.run_model()

                    TSFC_CRZ = prob.get_val("CRZ.perf.TSFC")
                    TSFC_TOC = prob.get_val("TOC.perf.TSFC")
                    TSFC_RTO = prob.get_val("RTO.perf.TSFC")
                    TSFC_SLS = prob.get_val("SLS.perf.TSFC")
                    TSEC_CRZ = prob.get_val("CRZ.tsec_perf.TSEC")
                    TSEC_TOC = prob.get_val("TOC.tsec_perf.TSEC")
                    TSEC_RTO = prob.get_val("RTO.tsec_perf.TSEC")
                    TSEC_SLS = prob.get_val("SLS.tsec_perf.TSEC")
                    TOC_mdot = prob.get_val("TOC.inject.mix:W")
                    CRZ_mdot = prob.get_val("CRZ.inject.mix:W")

                except om.AnalysisError:
                    print("\n\n===== Error, continuing =====\n\n")

                    TSFC_CRZ = 0.0
                    TSFC_TOC = 0.0
                    TSFC_RTO = 0.0
                    TSFC_SLS = 0.0
                    TSEC_CRZ = 0.0
                    TSEC_TOC = 0.0
                    TSEC_RTO = 0.0
                    TSEC_SLS = 0.0
                    TOC_mdot = 0.0
                    CRZ_mdot = 0.0
                    break

                if use_h2:
                    fuel = "H2"
                else:
                    fuel = "JetA"
                fname = "../OUTPUT/N3_trends/N3_sweeps/" + fuel + f"/TOC-{i}_CRZ-{j}.pkl"

                with open(fname, "wb") as f:
                    pkl.dump(
                        np.vstack(
                            (
                                TSFC_TOC,
                                TSFC_RTO,
                                TSFC_SLS,
                                TSFC_CRZ,
                                TSEC_TOC,
                                TSEC_RTO,
                                TSEC_SLS,
                                TSEC_CRZ,
                                TOC_mdot,
                                CRZ_mdot,
                                TOCw,
                                CRZw,
                            )
                        ),
                        f,
                    )
            count += 1

    print("time", time.time() - st)
