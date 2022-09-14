#!/usr/bin/env python
"""
@File    :   hbtf_opt.py
@Time    :   2022/02/15
@Desc    :   None
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import time
import os

# ==============================================================================
# External Python modules
# ==============================================================================
import openmdao.api as om
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
from hbtf_HXnozz import viewer
from mp_hbtf_HXnozz import MPHBTF
import constants.constants as con

# from components.misc_components import area_con


def opt_prob(dPqP, output_dir):
    prob = om.Problem()
    prob.model = MPHBTF()

    # ==============================================================================
    # Optimizer setup
    # ==============================================================================
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options["optimizer"] = "SNOPT"
    prob.driver.options["debug_print"] = ["desvars", "nl_cons", "objs"]

    prob.driver.opt_settings["LU complete pivoting"] = None
    prob.driver.opt_settings["Hessian full memory"] = None
    prob.driver.opt_settings["Major feasibility tolerance"] = 1e-8
    prob.driver.opt_settings["Major optimality tolerance"] = 1e-3
    prob.driver.opt_settings["Penalty parameter"] = 1.0
    # prob.driver.opt_settings["Major step limit"] = 0.1
    prob.driver.opt_settings["Print file"] = os.path.join(output_dir, "SNOPT_print.out")
    prob.driver.opt_settings["Summary file"] = os.path.join(output_dir, "SNOPT_summary.out")
    prob.driver.opt_settings["Verify level"] = 0

    # ==============================================================================
    # Problem Recording
    # ==============================================================================
    prob.driver.hist_file = os.path.join(output_dir, f"history_{dPqP}.out")

    # ==============================================================================
    # Design variables
    # ==============================================================================
    # HX Geometry Variables
    prob.model.add_design_var("TOC.hx.channel_width_cold", lower=1.0, units="mm")
    prob.model.add_design_var("TOC.hx.channel_height_cold", lower=1.0, units="mm")
    prob.model.add_design_var("TOC.hx.fin_length_cold", lower=0.1, units="mm")
    # prob.model.add_design_var("TOC.hx.fin_thickness", lower=0.01, units="mm")
    prob.model.add_design_var("TOC.hx.mdot_hot", lower=0.01, units="kg/s")
    # prob.model.add_design_var("TOC.hx.T_in_hot", lower=273.0, units="degK")

    # Engine Parameters
    prob.model.add_design_var("TOC.T4_MAX", upper=3200.0, units="degR")
    prob.model.add_design_var("TOC.byp_splitter.BPR", lower=4.0)  # lower=4
    # prob.model.add_design_var("TOC.fan.PR", lower=1.2, upper=1.8)  # lower=1.1, upper=2.0
    # prob.model.add_design_var("TOC.lpc.PR", lower=1.5, upper=3.8)  # lower=1.01, upper=4.05
    # prob.model.add_design_var("TOC.hpc.PR", lower=2.0, upper=20.0)  # lower=1.01, upper=20.0

    # ==============================================================================
    # Constraints
    # ==============================================================================
    # prob.model.add_constraint("TOC.fan.map.NcMap", upper=1.0)
    prob.model.add_constraint("TOC.HXduct.dPqP", upper=dPqP)
    prob.model.add_constraint("TOC.HX_area_con.area_con", equals=1.0)
    prob.model.add_constraint("TOC.heatcomp.T_out", upper=600.0, units="degK")
    # prob.model.add_constraint("TOC.hx.heat_transfer", equals=20000, units="W")
    prob.model.add_constraint("TOC.fan_dia", upper=100, units="inch")

    # ==============================================================================
    # Objective
    # ==============================================================================
    prob.model.add_objective("TOC.perf.TSFC")

    prob.setup(force_alloc_complex=True)

    # --- Design point inputs ---
    prob.set_val("TOC.splitter.BPR", 5.105)
    prob.set_val("TOC.fan.PR", 1.685)
    prob.set_val("TOC.fan.eff", 0.8948)
    prob.set_val("TOC.lpc.PR", 1.935)
    prob.set_val("TOC.lpc.eff", 0.9243)
    prob.set_val("TOC.hpc.PR", 9.369)
    prob.set_val("TOC.hpc.eff", 0.8707)
    prob.set_val("TOC.hpt.eff", 0.8888)
    prob.set_val("TOC.lpt.eff", 0.8996)
    prob.set_val("TOC.fc.alt", 37000.0, units="ft")
    prob.set_val("TOC.fc.MN", 0.78)
    prob.set_val("TOC.T4_MAX", 2857, units="degR")
    prob.set_val("TOC.Fn_DES", 5900.0, units="lbf")

    # --- HX fixed inputs ---
    hx_params = [
        ["case_thickness", 2.0, "mm"],
        ["fin_thickness", 0.102, "mm"],
        ["plate_thickness", 0.2, "mm"],
        ["material_k", con.k_al, "W/m/K"],
        ["material_rho", con.rho_al, "kg/m**3"],
        ["channel_height_cold", 10, "mm"],
        ["channel_width_cold", 10, "mm"],
        ["fin_length_cold", 6.0, "mm"],
        ["channel_height_hot", 1, "mm"],
        ["channel_height_hot", 1, "mm"],
        ["n_wide_cold", 50, None],
        ["n_long_cold", 3, None],
        ["n_tall", 1, None],
        ["fin_length_hot", 6.0, "mm"],
        ["cp_hot", con.cp_oil, "J/kg/K"],
        ["k_hot", con.k_oil, "W/m/K"],
        ["mu_hot", con.mu_oil, "kg/m/s"],
    ]

    for column in hx_params:
        prob.set_val(f"TOC.hx.{column[0]}", column[1], units=column[2])

    # --- Design point initial guesses ---

    prob["TOC.hx.channel_height_cold"] = 10.0
    prob["TOC.hx.channel_width_cold"] = 10.0
    prob["TOC.hx.mdot_hot"] = 0.1
    prob["TOC.hx.fin_length_cold"] = 6.0
    prob["TOC.core_nozz.PR"] = 8.0
    prob["TOC.splitter.BPR"] = 5.105
    prob["TOC.balance.W"] = 320.0
    prob["TOC.balance.FAR"] = 0.025

    # st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    # om.n2(prob)

    # prob.model.TOC.nonlinear_solver.options["maxiter"] = 0
    # prob.model.RTO.nonlinear_solver.options["maxiter"] = 0
    # prob.model.SLS.nonlinear_solver.options["maxiter"] = 0
    # prob.model.CRZ.nonlinear_solver.options["maxiter"] = 0

    # prob.run_model()
    # prob.check_totals(compact_print=True)

    prob.run_driver()

    # with open(f"{output_dir}_debug.out", "w") as file:
    #     prob.model.TOC.list_inputs(prom_name=True, units=True, out_stream=file)
    #     print("\n\n\n", file=file)
    #     prob.model.TOC.list_outputs(prom_name=True, units=True, out_stream=file)

    # with open("outputs.out", "w") as file:
    #     prob.model.list_outputs(prom_name=True, units=True, out_stream=file)

    # with open("inputs.out", "w") as file:
    #     prob.model.list_inputs(prom_name=True, units=True, out_stream=file)

    # with open(f"{output_dir}_output.txt", "w") as file:
    #     for pt in ["TOC"]:
    #         viewer(prob, pt, file)

    #     print(file=file, flush=True)
    #     print("Run time", time.time() - st, file=file, flush=True)

    # for pt in ["TOC"] + mp_hbtf.od_pts:
    #     map_plots(prob, pt)

    # prob.check_partials(compact_print=True, show_only_incorrect=True)


def run_mp_hbtf(value):
    prob = om.Problem()
    prob.model = MPHBTF()
    prob.setup()

    print("Test value is: ", value)
    prob.set_val("TOC.byp_splitter.BPR", value)
    prob.set_val("TOC.fan.PR", 1.685)
    prob.set_val("TOC.fan.eff", 0.8948)
    prob.set_val("TOC.lpc.PR", 1.935)
    prob.set_val("TOC.lpc.eff", 0.9243)
    prob.set_val("TOC.hpc.PR", 9.369)
    prob.set_val("TOC.hpc.eff", 0.8707)
    prob.set_val("TOC.hpt.eff", 0.8888)
    prob.set_val("TOC.lpt.eff", 0.8996)
    prob.set_val("TOC.fc.alt", 37000.0, units="ft")
    prob.set_val("TOC.fc.MN", 0.78)
    prob.set_val("TOC.T4_MAX", 2857, units="degR")
    prob.set_val("TOC.Fn_DES", 5900.0, units="lbf")

    # --- HX fixed inputs ---
    hx_params = [
        ["case_thickness", 2.0, "mm"],
        ["fin_thickness", 0.102, "mm"],
        ["plate_thickness", 0.2, "mm"],
        ["material_k", con.k_al, "W/m/K"],
        ["material_rho", con.rho_al, "kg/m**3"],
        ["channel_height_cold", 6.0, "mm"],
        ["channel_width_cold", 6.0, "mm"],
        ["fin_length_cold", 6.0, "mm"],
        ["channel_height_hot", 1, "mm"],
        ["channel_height_hot", 1, "mm"],
        ["n_wide_cold", 50, None],
        ["n_long_cold", 3, None],
        ["n_tall", 1, None],
        ["fin_length_hot", 6.0, "mm"],
        ["cp_hot", con.cp_oil, "J/kg/K"],
        ["k_hot", con.k_oil, "W/m/K"],
        ["mu_hot", con.mu_oil, "kg/m/s"],
    ]

    for column in hx_params:
        prob.set_val(f"TOC.hx.{column[0]}", column[1], units=column[2])

    # --- Design point initial guesses ---
    prob["TOC.core_nozz.PR"] = 8.0
    prob["TOC.balance.W"] = 320.0
    prob["TOC.balance.FAR"] = 0.025

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    prob.run_model()


if __name__ == "__main__":

    opt_prob(0.2, "OUTPUT")
