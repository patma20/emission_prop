#!/usr/bin/env python
"""
@File    :   N3_HXnozz_Opt.py
@Time    :   2022/04/05
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

# ==============================================================================
# Extension modules
# ==============================================================================
from N3_HXnozz_withHX import viewer, MPN3, map_plots
import constants.constants as con

# from components.misc_components import area_con


def opt_prob(dPqP, output_dir, elec_load=20, BPR=300, mdot_hot=1, save_res=True):
    prob = om.Problem()
    prob.model = MPN3()

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
    # prob.driver.opt_settings["Major step limit"] = 0.1
    if save_res is True:
        if os.path.isdir(output_dir) is False:
            os.mkdir(output_dir)
        prob.driver.opt_settings["Print file"] = os.path.join(output_dir, "SNOPT_print.out")
        prob.driver.opt_settings["Summary file"] = os.path.join(output_dir, "SNOPT_summary.out")
    prob.driver.opt_settings["Verify level"] = 3

    # ==============================================================================
    # Problem Recording
    # ==============================================================================
    if save_res is True:
        prob.driver.hist_file = os.path.join(output_dir, f"history_{elec_load}_BPR{BPR}.out")

    # ==============================================================================
    # Design variables
    # ==============================================================================
    # HX Geometry Variables
    prob.model.add_design_var("TOC.hx.channel_width_cold", lower=0.01, ref=10.0, units="mm")  # , ref=28.0
    prob.model.add_design_var("TOC.hx.channel_height_cold", lower=0.01, ref=10.0, units="mm")  # , ref=28.0
    # prob.model.add_design_var("TOC.hx.channel_width_hot", lower=0.01, ref=1.0, units="mm")  # , ref=28.0
    # prob.model.add_design_var("TOC.hx.channel_height_hot", lower=0.01, ref=1.0, units="mm")  # , ref=28.0
    prob.model.add_design_var("TOC.hx.fin_length_cold", lower=0.1, ref=6.0, units="mm")  # , ref=10
    # prob.model.add_design_var("TOC.hx.fin_thickness", lower=0.05, ref=0.102, units="mm")
    # prob.model.add_design_var("TOC.mdot_coolant", lower=0.01, units="kg/s")

    # Engine Parameters
    # prob.model.add_design_var("TOC.byp_splitter.BPR", upper=350.0, ref=BPR)  # lower=4
    # prob.model.add_design_var("TOC.splitter.BPR", lower=2, ref=23.94514401)  # lower=4
    prob.model.add_design_var("fan:PRdes", lower=1.20, upper=1.4)
    prob.model.add_design_var("lpc:PRdes", lower=2.0, upper=4.0)
    prob.model.add_design_var("TOC.balance.rhs:hpc_PR", lower=40.0, upper=70.0, ref0=40.0, ref=70.0)
    prob.model.add_design_var("TOC.balance.rhs:FAR", lower=3000.0, upper=3600.0, ref0=3000.0, ref=3600.0)

    # ==============================================================================
    # Constraints
    # ==============================================================================
    prob.model.add_constraint("TOC.HXduct.dPqP", upper=dPqP)
    prob.model.add_constraint("TOC.HX_area_con.area_con", equals=1.0)
    prob.model.add_constraint("TOC.heatcomp.T_out", upper=250.0, ref=300.0, units="degC")
    # prob.model.add_constraint("TOC.fan_dia.FanDia", upper=100.0, ref=100.0)
    prob.model.add_constraint("TOC.perf.Fn", lower=5800.0, ref=6000.0)

    # ==============================================================================
    # Objective
    # ==============================================================================
    prob.model.add_objective("TOC.perf.TSFC")

    prob.setup(force_alloc_complex=True)

    # --- HX fixed inputs ---
    hx_params = [
        ["case_thickness", 2.0, "mm"],
        ["fin_thickness", 0.102, "mm"],
        ["plate_thickness", 0.2, "mm"],
        ["material_k", con.k_al, "W/m/K"],
        ["material_rho", con.rho_al, "kg/m**3"],
        ["channel_height_cold", 10.0, "mm"],
        ["channel_width_cold", 10.0, "mm"],
        ["fin_length_cold", 6.0, "mm"],
        ["n_wide_cold", 50, None],
        ["n_long_cold", 3, None],
        ["n_tall", 15, None],
        ["channel_height_hot", 1.0, "mm"],
        ["channel_width_hot", 1.0, "mm"],
        ["fin_length_hot", 6.0, "mm"],
        ["cp_hot", con.cp_oil, "J/kg/K"],
        ["k_hot", con.k_oil, "W/m/K"],
        ["mu_hot", con.mu_oil, "kg/m/s"],
    ]

    for column in hx_params:
        prob.set_val(f"TOC.hx.{column[0]}", column[1], units=column[2])

    # --- Design point initial guesses ---
    prob.set_val("TOC.fc.W", 820.44097898, units="lbm/s")
    prob.set_val("TOC.splitter.BPR", 23.94514401)
    prob.set_val("TOC.byp_splitter.BPR", BPR)
    prob.set_val("TOC.balance.rhs:hpc_PR", 70.0)  # 53.6332

    # Set up the specific cycle parameters
    prob.set_val("fan:PRdes", 1.300)
    prob.set_val("lpc:PRdes", 4.000)  # 3
    prob.set_val("TOC.balance.rhs:FAR", 3400.0, units="degR")
    prob.set_val("TOC.thermal_params.heat_load", elec_load, units="kW")
    prob.set_val("TOC.mdot_coolant", mdot_hot, units="kg/s")

    # Set initial guesses for balances
    prob["TOC.balance.FAR"] = 0.02650
    prob["TOC.balance.lpt_PR"] = 10.937
    prob["TOC.balance.hpt_PR"] = 4.185
    prob["TOC.fc.balance.Pt"] = 5.272
    prob["TOC.fc.balance.Tt"] = 444.41

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    # om.n2(prob)

    # prob.run_model()
    # prob.check_partials(compact_print=True, show_only_incorrect=True, method="fd", excludes="*air*")
    # prob.check_totals(
    # of=["TOC.perf.TSFC"],
    # wrt=["TOC.hx.channel_width_cold", "TOC.hx.channel_height_cold", "TOC.hx.fin_length_cold"],
    # compact_print=True,
    # method="fd",
    # step=1e-2,
    # step_calc="rel_avg",
    # )

    prob.run_driver()

    # with open(f"{output_dir}_debug.out", "w") as file:
    #     prob.model.TOC.list_inputs(prom_name=True, units=True, out_stream=file)
    #     print("\n\n\n", file=file)
    #     prob.model.TOC.list_outputs(prom_name=True, units=True, out_stream=file)

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

    map_plots(prob, "TOC")

    return prob.get_val("TOC.perf.TSFC")


if __name__ == "__main__":
    elec_load = 50  # kW
    mdot_hot = 1
    opt_prob(
        dPqP=0.25,
        elec_load=elec_load,
        mdot_hot=mdot_hot,
        output_dir=f"../OUTPUT/N3_opt_{mdot_hot}kgs_{elec_load}kW_full",
        save_res=True,
    )
