# --- Python 3.8 ---
"""
HBTF multi-point model in pyCycle
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import time

# ==============================================================================
# External Python modules
# ==============================================================================
import pycycle.api as pyc
import openmdao.api as om

# ==============================================================================
# Extension modules
# ==============================================================================
import constants.constants as con

if __name__ == "__main__":
    from hbtf import HBTF, viewer, map_plots
else:
    from .hbtf import HBTF, viewer, map_plots


class MPHBTF(pyc.MPCycle):
    def setup(self):

        self.pyc_add_pnt("TOC", HBTF())  # Create an instance of the High Bypass ratio Turbofan

        # --- Setup TOC point ---
        self.set_input_defaults("TOC.inlet.MN", 0.751)
        self.set_input_defaults("TOC.fan.MN", 0.4578)
        self.set_input_defaults("TOC.splitter.BPR", 5.105)
        self.set_input_defaults("TOC.splitter.MN1", 0.3104)
        self.set_input_defaults("TOC.splitter.MN2", 0.4518)
        self.set_input_defaults("TOC.duct4.MN", 0.3121)
        self.set_input_defaults("TOC.lpc.MN", 0.3059)
        self.set_input_defaults("TOC.duct6.MN", 0.3563)
        self.set_input_defaults("TOC.hpc.MN", 0.2442)
        self.set_input_defaults("TOC.bld3.MN", 0.3000)
        self.set_input_defaults("TOC.burner.MN", 0.1025)
        self.set_input_defaults("TOC.hpt.MN", 0.3650)
        self.set_input_defaults("TOC.duct11.MN", 0.3063)
        self.set_input_defaults("TOC.lpt.MN", 0.4127)
        self.set_input_defaults("TOC.duct13.MN", 0.4463)
        self.set_input_defaults("TOC.byp_bld.MN", 0.4489)
        self.set_input_defaults("TOC.duct15.MN", 0.4589)
        self.set_input_defaults("TOC.LP_Nmech", 4666.1, units="rpm")
        self.set_input_defaults("TOC.HP_Nmech", 14705.7, units="rpm")
        self.set_input_defaults("TOC.hx.cp_hot", 2500, units="J/kg/K")

        # --- Set up bleed values -----
        self.pyc_add_cycle_param("inlet.ram_recovery", 0.9990)
        self.pyc_add_cycle_param("duct4.dPqP", 0.0048)
        self.pyc_add_cycle_param("duct6.dPqP", 0.0101)
        self.pyc_add_cycle_param("burner.dPqP", 0.0540)
        self.pyc_add_cycle_param("duct11.dPqP", 0.0051)
        self.pyc_add_cycle_param("duct13.dPqP", 0.0107)
        # self.pyc_add_cycle_param("duct15.dPqP", 0.0149)
        self.pyc_add_cycle_param("core_nozz.Cv", 0.9933)
        self.pyc_add_cycle_param("byp_bld.bypBld:frac_W", 0.005)
        self.pyc_add_cycle_param("byp_nozz.Cv", 0.9939)
        self.pyc_add_cycle_param("hpc.cool1:frac_W", 0.050708)
        self.pyc_add_cycle_param("hpc.cool1:frac_P", 0.5)
        self.pyc_add_cycle_param("hpc.cool1:frac_work", 0.5)
        self.pyc_add_cycle_param("hpc.cool2:frac_W", 0.020274)
        self.pyc_add_cycle_param("hpc.cool2:frac_P", 0.55)
        self.pyc_add_cycle_param("hpc.cool2:frac_work", 0.5)
        self.pyc_add_cycle_param("bld3.cool3:frac_W", 0.067214)
        self.pyc_add_cycle_param("bld3.cool4:frac_W", 0.101256)
        self.pyc_add_cycle_param("hpc.cust:frac_P", 0.5)
        self.pyc_add_cycle_param("hpc.cust:frac_work", 0.5)
        self.pyc_add_cycle_param("hpc.cust:frac_W", 0.0445)
        self.pyc_add_cycle_param("hpt.cool3:frac_P", 1.0)
        self.pyc_add_cycle_param("hpt.cool4:frac_P", 0.0)
        self.pyc_add_cycle_param("lpt.cool1:frac_P", 1.0)
        self.pyc_add_cycle_param("lpt.cool2:frac_P", 0.0)
        self.pyc_add_cycle_param("hp_shaft.HPX", 250.0, units="hp")

        idv = self.add_subsystem("idv", om.IndepVarComp(), promotes=["*"])
        idv.add_output("T4_max", val=3600.0, units="degR")

        self.od_pts = ["RTO", "SLS", "CRZ"]

        self.od_MNs = [0.25, 0.000001, 0.78]
        self.od_alts = [0.0, 0.0, 37000.0]
        self.od_dTs = [0.0, 0.0, 0.0]

        bal = prob.model.add_subsystem("bal", om.BalanceComp(), promotes=["RTO_T4"])
        bal.add_balance("TOC_BPR", val=5.0, units=None, eq_units="ft/s", use_mult=True)
        prob.model.connect("bal.TOC_BPR", "TOC.splitter.BPR")
        prob.model.connect("CRZ.byp_nozz.Fl_O:stat:V", "bal.lhs:TOC_BPR")
        prob.model.connect("CRZ.core_nozz.Fl_O:stat:V", "bal.rhs:TOC_BPR")

        bal.add_balance("TOC_W", val=320.0, units="lbm/s", eq_units="degR", rhs_name="RTO_T4")
        prob.model.connect("bal.TOC_W", "TOC.fc.W")
        prob.model.connect("RTO.burner.Fl_O:tot:T", "bal.lhs:TOC_W")

        self.add_subsystem(
            "T4_ratio",
            om.ExecComp(
                "TOC_T4 = RTO_T4*TR",
                RTO_T4={"val": 3400.0, "units": "degR"},
                TOC_T4={"val": 3150.0, "units": "degR"},
                TR={"val": 0.92, "units": None},
            ),
            promotes_inputs=["RTO_T4"],
        )
        for i, pt in enumerate(self.od_pts):
            if pt == "RTO":
                self.pyc_add_pnt(f"{pt}", HBTF(design=False, constrained_balance=False))
            else:
                self.pyc_add_pnt(f"{pt}", HBTF(design=False))

            self.set_input_defaults(f"{pt}.fc.MN", self.od_MNs[i])
            self.set_input_defaults(f"{pt}.fc.alt", self.od_alts[i], units="ft")
            self.set_input_defaults(f"{pt}.fc.dTs", self.od_dTs[i], units="degR")

            if pt in ["CRZ", "SLS"]:
                self.connect("T4_max", pt + ".FAR.T4max")
                self.set_input_defaults(pt + ".FAR.NcMapTgt", val=1.0)
                self.set_input_defaults(pt + ".FAR.T3max", val=1750.0, units="degR")

        self.connect("T4_ratio.TOC_T4", "TOC.balance.rhs:FAR")

        self.set_input_defaults("RTO_T4", val=2900.0, units="degR")
        self.set_input_defaults("RTO.balance.rhs:FAR", val=16000.0, units="lbf")

        self.pyc_use_default_des_od_conns()

        # Set up the RHS of the balances!
        self.pyc_connect_des_od("core_nozz.Throat:stat:area", "balance.rhs:W")
        self.pyc_connect_des_od("byp_nozz.Throat:stat:area", "balance.rhs:BPR")

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options["atol"] = 1e-8
        newton.options["rtol"] = 1e-99
        newton.options["iprint"] = 2
        newton.options["maxiter"] = 7
        newton.options["solve_subsystems"] = True
        newton.options["max_sub_solves"] = 100
        newton.options["err_on_non_converge"] = False

        newton.linesearch = om.ArmijoGoldsteinLS()
        newton.linesearch.options["bound_enforcement"] = "scalar"
        newton.linesearch.options["iprint"] = -1
        self.linear_solver = om.DirectSolver(assemble_jac=True)

        super().setup()


if __name__ == "__main__":

    prob = om.Problem()

    prob.model = mp_hbtf = MPHBTF()

    prob.setup(force_alloc_complex=True)

    # --- Design point inputs ---
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

    # --- HX fixed inputs ---
    hx_params = [
        ["case_thickness", 2.0, "mm"],
        ["fin_thickness", 0.102, "mm"],
        ["plate_thickness", 0.2, "mm"],
        ["material_k", con.k_al, "W/m/K"],
        ["material_rho", con.rho_al, "kg/m**3"],
        ["channel_height_cold", 14, "mm"],
        ["channel_width_cold", 1.35, "mm"],
        ["fin_length_cold", 6.0, "mm"],
        ["channel_height_hot", 1, "mm"],
        ["channel_height_hot", 1, "mm"],
        ["fin_length_hot", 6.0, "mm"],
        ["cp_hot", con.cp_oil, "J/kg/K"],
        ["k_hot", con.k_oil, "W/m/K"],
        ["mu_hot", con.mu_oil, "kg/m/s"],
    ]

    for column in hx_params:
        prob.set_val(f"TOC.hx.{column[0]}", column[1], units=column[2])

    # --- Design point initial guesses ---
    # prob["TOC.balance.W"] = 340.0
    prob["TOC.balance.FAR"] = 0.025

    # --- Initial guesses off-design points ---
    for pt in mp_hbtf.od_pts:
        # initial guesses
        if pt not in ["SLS", "CRZ"]:
            prob[f"{pt}.balance.FAR"] = 0.02467
        prob[f"{pt}.balance.W"] = 300
        prob[f"{pt}.balance.BPR"] = 5.105
        prob[f"{pt}.balance.lp_Nmech"] = 5000
        prob[f"{pt}.balance.hp_Nmech"] = 15000
        prob[f"{pt}.hpt.PR"] = 3.0
        prob[f"{pt}.lpt.PR"] = 4.0
        prob[f"{pt}.fan.map.RlineMap"] = 2.0
        prob[f"{pt}.lpc.map.RlineMap"] = 2.0
        prob[f"{pt}.hpc.map.RlineMap"] = 2.0

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    # om.n2(prob)

    prob.run_model()
    # with open("outputs.out", "w") as file:
    #     prob.model.list_outputs(prom_name=True, units=True, out_stream=file)

    # with open("inputs.out", "w") as file:
    #     prob.model.list_inputs(prom_name=True, units=True, out_stream=file)

    # prob.check_partials(compact_print=True, show_only_incorrect=True, excludes=["*air*"])

    # print(prob.get_val("TOC.hx.cp_cold"))
    # print(prob.get_val("TOC.duct15.Fl_O:stat:Cp"))
    # print(prob.get_val("TOC.duct15.Fl_O:stat:"))

    # print(prob.get_val("TOC.hx.k_cold"))
    # print(prob.get_val("TOC.air.k"))

    with open("hbtf_output1.txt", "w") as file:
        for pt in ["TOC"] + mp_hbtf.od_pts:
            viewer(prob, pt, file)

        print(file=file, flush=True)
        print("Run time", time.time() - st, file=file, flush=True)

    for pt in ["TOC"] + mp_hbtf.od_pts:
        map_plots(prob, pt)
