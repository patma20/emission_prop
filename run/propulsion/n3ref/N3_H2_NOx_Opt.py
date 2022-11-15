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
import pickle

# ==============================================================================
# External Python modules
# ==============================================================================
import openmdao.api as om
import pycycle.api as pyc

# ==============================================================================
# Extension modules
# ==============================================================================
from N3ref import N3, viewer, MPN3


def N3_MDP_Opt_model(output_dir, save_res=False):

    prob = om.Problem()
    prob.model = MPN3(use_h2=True)  # order_add=["bal"]

    prob.model.pyc_add_cycle_param("ext_ratio.core_Cv", 0.9999)
    prob.model.pyc_add_cycle_param("ext_ratio.byp_Cv", 0.9975)
    # prob.model.pyc_add_cycle_param("EINOx.H_SLS", 0.007)

    # bal = prob.model.add_subsystem("bal", om.BalanceComp(), promotes=["RTO_T4"])

    # bal.add_balance("TOC_BPR", val=23.7281, units=None, eq_units=None)
    # prob.model.connect("bal.TOC_BPR", "TOC.splitter.BPR")
    # prob.model.connect("CRZ.ext_ratio.ER", "bal.lhs:TOC_BPR")

    # bal.add_balance("TOC_W", val=820.95, units="lbm/s", eq_units="degR", rhs_name="RTO_T4")
    # prob.model.connect("bal.TOC_W", "TOC.fc.W")
    # prob.model.connect("RTO.burner.Fl_O:tot:T", "bal.lhs:TOC_W")

    # bal.add_balance(
    #     "CRZ_Fn_target", val=5514.4, units="lbf", eq_units="lbf", use_mult=True, mult_val=0.9, ref0=5000.0, ref=7000.0
    # )
    # prob.model.connect("bal.CRZ_Fn_target", "CRZ.balance.rhs:FAR")
    # prob.model.connect("TOC.perf.Fn", "bal.lhs:CRZ_Fn_target")
    # prob.model.connect("CRZ.perf.Fn", "bal.rhs:CRZ_Fn_target")

    # bal.add_balance(
    #     "SLS_Fn_target",
    #     val=28620.8,
    #     units="lbf",
    #     eq_units="lbf",
    #     use_mult=True,
    #     mult_val=1.2553,
    #     ref0=28000.0,
    #     ref=30000.0,
    # )
    # prob.model.connect("bal.SLS_Fn_target", "SLS.balance.rhs:FAR")
    # prob.model.connect("RTO.perf.Fn", "bal.lhs:SLS_Fn_target")
    # prob.model.connect("SLS.perf.Fn", "bal.rhs:SLS_Fn_target")

    # ==============================================================================
    # Optimizer setup
    # ==============================================================================
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options["optimizer"] = "SNOPT"
    prob.driver.options["debug_print"] = ["desvars", "nl_cons", "objs"]

    prob.driver.opt_settings["LU complete pivoting"] = None
    prob.driver.opt_settings["Hessian full memory"] = None
    prob.driver.opt_settings["Major feasibility tolerance"] = 1e-8
    prob.driver.opt_settings["Major optimality tolerance"] = 1e-8
    prob.driver.opt_settings["Penalty parameter"] = 1.0
    prob.driver.opt_settings["Major step limit"] = 0.1

    modelname = "H2"

    if save_res is True:
        if os.path.isdir(output_dir) is False:
            os.mkdir(output_dir)
        prob.driver.opt_settings["Print file"] = os.path.join(output_dir, "SNOPT_print_" + modelname + ".out")
        prob.driver.opt_settings["Summary file"] = os.path.join(output_dir, "SNOPT_summary_" + modelname + ".out")
    # prob.driver.opt_settings["Verify level"] = 3

    # ==============================================================================
    # Problem Recording
    # ==============================================================================
    if save_res is True:
        prob.driver.hist_file = os.path.join(output_dir, "history_" + modelname + ".out")

    # ==============================================================================
    # Design variables
    # ==============================================================================
    # prob.model.add_design_var("fan:PRdes", lower=1.2, upper=1.3)
    prob.model.add_design_var("lpc:PRdes", lower=2.5, upper=8.0)
    # prob.model.add_design_var("TOC.balance.rhs:hpc_PR", lower=40.0, upper=70.0, ref0=40.0, ref=70.0)
    prob.model.add_design_var("RTO_T4", lower=3000.0, upper=3600.0, ref0=3000.0, ref=3600.0)
    # prob.model.add_design_var("bal.rhs:TOC_BPR", lower=1.35, upper=1.45, ref0=1.35, ref=1.45)
    # prob.model.add_design_var("TOC.splitter.BPR", lower=20, upper=30, ref0=20, ref=30)
    prob.model.add_design_var("T4_ratio.TR", lower=0.5, upper=0.95, ref0=0.5, ref=0.95)

    # ==============================================================================
    # Constraints
    # ==============================================================================
    # prob.model.add_constraint("TOC.fan_dia.FanDia", upper=100.0, ref=100.0)
    prob.model.add_constraint("TOC.perf.Fn", lower=5800.0, ref=6000.0)
    # prob.model.add_constraint("TOC.NOx.EINOx", upper=18.0, ref=18.0)

    # ==============================================================================
    # Objective
    # ==============================================================================
    prob.model.add_objective("CRZ.perf.TSFC", ref0=0.4, ref=0.5)

    prob.model.set_input_defaults("RTO_T4", 3400.0, units="degR")

    return prob


if __name__ == "__main__":
    save_res = True
    output_dir = "../OUTPUT/N3_opt"
    prob = N3_MDP_Opt_model(output_dir, save_res)

    prob.setup()

    # Define the design point
    prob.set_val("TOC.fc.W", 820.44097898, units="lbm/s")
    prob.set_val("TOC.splitter.BPR", 23.94514401)
    prob.set_val("TOC.balance.rhs:hpc_PR", 53.6332)

    # Set specific cycle parameters
    prob.set_val("SLS.fc.MN", 0.001)
    prob.set_val("SLS.balance.rhs:FAR", 28620.84, units="lbf")
    prob.set_val("CRZ.balance.rhs:FAR", 5510.72833567, units="lbf")
    # prob.set_val("bal.rhs:TOC_BPR", 1.40)
    prob.set_val("T4_ratio.TR", 0.926470588)
    prob.set_val("RTO_T4", 3400.0, units="degR")
    prob.set_val("fan:PRdes", 1.300)
    prob.set_val("lpc:PRdes", 3.000)
    prob.set_val("RTO.hpt_cooling.x_factor", 0.9)

    # Set inital guesses for balances
    prob["TOC.balance.FAR"] = 0.02650
    # prob["bal.TOC_W"] = 820.44097898
    prob["TOC.balance.lpt_PR"] = 10.937
    prob["TOC.balance.hpt_PR"] = 4.185
    prob["TOC.fc.balance.Pt"] = 5.272
    prob["TOC.fc.balance.Tt"] = 444.41

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

    st = time.time()

    # N2 generation
    # om.n2(prob)

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    # prob.run_model()
    # prob.check_partials(compact_print=True, show_only_incorrect=False, method="fd")
    # prob.check_totals(
    #     of=["CRZ.NOx.EINOx"],
    #     # wrt=["CRZ.NOx.P3", "CRZ.NOx.T3", "CRZ.NOx.T4"],
    #     wrt=["RTO_T4", "bal.rhs:TOC_BPR"],
    #     compact_print=True,
    #     method="fd",
    #     step=1e-6,
    #     step_calc="rel_avg",
    # )

    prob.run_driver()

    # for pt in ["TOC"] + prob.model.od_pts:
    #     viewer(prob, pt)

    if save_res is True:
        with open(f"{output_dir}/outputs.out", "w") as file:
            prob.model.list_outputs(prom_name=True, units=True, out_stream=file)

        with open(f"{output_dir}/inputs.out", "w") as file:
            prob.model.list_inputs(prom_name=True, units=True, out_stream=file)

        with open(f"{output_dir}/output.txt", "w") as file:
            for pt in ["TOC"]:
                viewer(prob, pt, file)

            print(file=file, flush=True)
            print("Run time", time.time() - st, file=file, flush=True)

    # Create compressor and turbine maps
    # map_plots(prob, "TOC")

    # print("Diameter", prob["TOC.fan_dia.FanDia"][0])
    # print("ER", prob["CRZ.ext_ratio.ER"])
    # print("EINOx", prob["CRZ.NOx.EINOx"])
    # print("Composition", prob["CRZ.burner.Fl_O:tot:composition"])
    # print("Composition", prob["CRZ.burner.Fl_I:tot:composition"])

    print("time", time.time() - st)
