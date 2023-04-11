#!/usr/bin/env python
"""
@File    :   N3_vapor_rec.py
@Time    :   2022/11/22
@Desc    :   None
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import sys
import os

# ==============================================================================
# External Python modules
# ==============================================================================
import openmdao.api as om
import pycycle.api as pyc
import pickle as pkl
import numpy as np

# from mpi4py import MPI

# ==============================================================================
# Extension modules
# ==============================================================================
# import pycycle.constants as con

from components.emissions import EINOx, TSEC
from components.injector import Injector
from components.extractor import WaterBleed

from small_core_eff_balance import SmallCoreEffBalance

from N3_Fan_map import FanMap
from N3_LPC_map import LPCMap
from N3_HPC_map import HPCMap
from N3_HPT_map import HPTMap
from N3_LPT_map import LPTMap


class N3(pyc.Cycle):
    def initialize(self):
        self.options.declare("cooling", default=False, desc="If True, calculate cooling flow values.")
        self.options.declare("use_h2", default=False, types=bool, desc="If True, use hydrogen as the fuel.")
        self.options.declare("wet_air", default=False, types=bool, desc="If True, use wet air.")
        self.options.declare("design_water", default=False, types=bool, desc="If True, set DP as water injection.")

        super().initialize()

    def setup(self):

        USE_TABULAR = False

        if USE_TABULAR:
            self.options["thermo_method"] = "TABULAR"
            self.options["thermo_data"] = pyc.AIR_JETA_TAB_SPEC
            FUEL_TYPE = "FAR"
        else:
            self.options["thermo_method"] = "CEA"

            if self.options["wet_air"] is True:
                self.options["thermo_data"] = pyc.species_data.wet_air
            else:
                self.options["thermo_data"] = pyc.species_data.janaf

            if self.options["use_h2"]:
                FUEL_TYPE = "H2"
            else:
                FUEL_TYPE = "Jet-A(g)"

        cooling = self.options["cooling"]
        design = self.options["design"]
        design_water = self.options["design_water"]

        self.add_subsystem(
            "fc", pyc.FlightConditions(composition=pyc.CEA_AIR_COMPOSITION, reactant="Water", mix_ratio_name="WAR")
        )
        self.add_subsystem("inlet", pyc.Inlet())
        self.add_subsystem(
            "fan",
            pyc.Compressor(map_data=FanMap, map_extrap=True, bleed_names=[]),
            promotes_inputs=[("Nmech", "Fan_Nmech")],
        )
        self.add_subsystem("splitter", pyc.Splitter())
        self.add_subsystem("duct2", pyc.Duct(expMN=2.0))
        self.add_subsystem(
            "lpc", pyc.Compressor(map_data=LPCMap, map_extrap=True), promotes_inputs=[("Nmech", "LP_Nmech")]
        )
        self.add_subsystem("bld25", pyc.BleedOut(bleed_names=["sbv"]))
        self.add_subsystem("duct25", pyc.Duct(expMN=2.0))
        self.add_subsystem(
            "hpc",
            pyc.Compressor(map_data=HPCMap, map_extrap=True, bleed_names=["bld_inlet", "bld_exit", "cust"]),
            promotes_inputs=[("Nmech", "HP_Nmech")],
        )
        self.add_subsystem("bld3", pyc.BleedOut(bleed_names=["bld_inlet", "bld_exit"]))
        self.add_subsystem("burner", pyc.Combustor(fuel_type=FUEL_TYPE))
        self.add_subsystem(
            "hpt",
            pyc.Turbine(map_data=HPTMap, map_extrap=True, bleed_names=["bld_inlet", "bld_exit"]),
            promotes_inputs=[("Nmech", "HP_Nmech")],
        )
        self.add_subsystem("duct45", pyc.Duct(expMN=2.0))
        self.add_subsystem(
            "lpt",
            pyc.Turbine(map_data=LPTMap, map_extrap=True, bleed_names=["bld_inlet", "bld_exit"]),
            promotes_inputs=[("Nmech", "LP_Nmech")],
        )
        self.add_subsystem("duct5", pyc.Duct(expMN=2.0))
        self.add_subsystem("core_nozz", pyc.Nozzle(nozzType="CV", lossCoef="Cv"))

        self.add_subsystem("byp_bld", pyc.BleedOut(bleed_names=["bypBld"]))
        self.add_subsystem("duct17", pyc.Duct(expMN=2.0))
        self.add_subsystem("byp_nozz", pyc.Nozzle(nozzType="CV", lossCoef="Cv"))

        self.add_subsystem("fan_shaft", pyc.Shaft(num_ports=2), promotes_inputs=[("Nmech", "Fan_Nmech")])
        self.add_subsystem("gearbox", pyc.Gearbox(), promotes_inputs=[("N_in", "LP_Nmech"), ("N_out", "Fan_Nmech")])
        self.add_subsystem("lp_shaft", pyc.Shaft(num_ports=3), promotes_inputs=[("Nmech", "LP_Nmech")])
        self.add_subsystem("hp_shaft", pyc.Shaft(num_ports=2), promotes_inputs=[("Nmech", "HP_Nmech")])
        self.add_subsystem("perf", pyc.Performance(num_nozzles=2, num_burners=1))

        fuel_vars = self.add_subsystem("fuelVars", om.IndepVarComp(), promotes_outputs=["*"])
        if not self.options["use_h2"]:
            LHV = 18564  # Lower Heating Value of JetA in BTU/lb
        else:
            LHV = 51591  # Lower Heating Value of H2 in BTU/lb
        fuel_vars.add_output("fuel_lhv", val=LHV, units="Btu/lbm")
        self.add_subsystem("tsec_perf", TSEC())

        self.add_subsystem("extract", WaterBleed())  # design=design_water
        self.add_subsystem("inject", Injector(reactant="Water", mix_name="mix"))  # , design=design_water

        # Performance connections
        self.connect("inlet.Fl_O:tot:P", "perf.Pt2")
        self.connect("hpc.Fl_O:tot:P", "perf.Pt3")
        self.connect("burner.Wfuel", "perf.Wfuel_0")
        self.connect("inlet.F_ram", "perf.ram_drag")
        self.connect("core_nozz.Fg", "perf.Fg_0")
        self.connect("byp_nozz.Fg", "perf.Fg_1")
        self.connect("perf.TSFC", "tsec_perf.TSFC")
        self.connect("fuel_lhv", "tsec_perf.LHV")

        # Mechanical Connections
        self.connect("fan.trq", "fan_shaft.trq_0")
        self.connect("gearbox.trq_out", "fan_shaft.trq_1")
        self.connect("gearbox.trq_in", "lp_shaft.trq_0")
        self.connect("lpc.trq", "lp_shaft.trq_1")
        self.connect("lpt.trq", "lp_shaft.trq_2")
        self.connect("hpc.trq", "hp_shaft.trq_0")
        self.connect("hpt.trq", "hp_shaft.trq_1")
        self.connect("fc.Fl_O:stat:P", "core_nozz.Ps_exhaust")
        self.connect("fc.Fl_O:stat:P", "byp_nozz.Ps_exhaust")

        self.add_subsystem(
            "ext_ratio",
            om.ExecComp(
                "ER = core_V_ideal * core_Cv / ( byp_V_ideal *  byp_Cv )",
                core_V_ideal={"val": 1000.0, "units": "ft/s"},
                core_Cv={"val": 0.98, "units": None},
                byp_V_ideal={"val": 1000.0, "units": "ft/s"},
                byp_Cv={"val": 0.98, "units": None},
                ER={"val": 1.4, "units": None},
            ),
        )

        self.connect("core_nozz.ideal_flow.V", "ext_ratio.core_V_ideal")
        self.connect("byp_nozz.ideal_flow.V", "ext_ratio.byp_V_ideal")

        main_order = [
            "fc",
            "inlet",
            "fan",
            "splitter",
            "duct2",
            "lpc",
            "bld25",
            "duct25",
            "inject",
            "hpc",
            "bld3",
            "burner",
            "hpt",
            "duct45",
            "lpt",
            "duct5",
            "extract",
            "core_nozz",
            "byp_bld",
            "duct17",
            "byp_nozz",
            "gearbox",
            "fan_shaft",
            "lp_shaft",
            "hp_shaft",
            "perf",
            "fuelVars",
            "tsec_perf",
            "ext_ratio",
        ]

        balance = self.add_subsystem("balance", om.BalanceComp())
        if design:

            balance.add_balance("FAR", eq_units="degR", lower=1e-4, val=0.017)
            self.connect("balance.FAR", "burner.Fl_I:FAR")
            self.connect("burner.Fl_O:tot:T", "balance.lhs:FAR")

            balance.add_balance("lpt_PR", val=10.937, lower=1.001, upper=20, eq_units="hp", rhs_val=0.0, res_ref=1e4)
            self.connect("balance.lpt_PR", "lpt.PR")
            self.connect("lp_shaft.pwr_net", "balance.lhs:lpt_PR")

            balance.add_balance("hpt_PR", val=4.185, lower=1.001, upper=8, eq_units="hp", rhs_val=0.0, res_ref=1e4)
            self.connect("balance.hpt_PR", "hpt.PR")
            self.connect("hp_shaft.pwr_net", "balance.lhs:hpt_PR")

            balance.add_balance("gb_trq", val=23928.0, units="ft*lbf", eq_units="hp", rhs_val=0.0, res_ref=1e4)
            self.connect("balance.gb_trq", "gearbox.trq_base")
            self.connect("fan_shaft.pwr_net", "balance.lhs:gb_trq")

            balance.add_balance("hpc_PR", val=14.0, units=None, eq_units=None)
            self.connect("balance.hpc_PR", ["hpc.PR", "opr_calc.HPCPR"])
            # self.connect('perf.OPR', 'balance.lhs:hpc_PR')
            self.connect("opr_calc.OPR_simple", "balance.lhs:hpc_PR")

            balance.add_balance("fan_eff", val=0.9689, units=None, eq_units=None)
            self.connect("balance.fan_eff", "fan.eff")
            self.connect("fan.eff_poly", "balance.lhs:fan_eff")

            balance.add_balance("lpc_eff", val=0.8895, units=None, eq_units=None)
            self.connect("balance.lpc_eff", "lpc.eff")
            self.connect("lpc.eff_poly", "balance.lhs:lpc_eff")

            # balance.add_balance('hpc_eff', val=0.8470, units=None, eq_units=None)
            balance.add_balance("hpt_eff", val=0.9226, units=None, eq_units=None)
            self.connect("balance.hpt_eff", "hpt.eff")
            self.connect("hpt.eff_poly", "balance.lhs:hpt_eff")

            balance.add_balance("lpt_eff", val=0.9401, units=None, eq_units=None)
            self.connect("balance.lpt_eff", "lpt.eff")
            self.connect("lpt.eff_poly", "balance.lhs:lpt_eff")

            self.add_subsystem(
                "hpc_CS",
                om.ExecComp(
                    "CS = Win *(power(Tout/518.67,0.5)/(Pout/14.696))",
                    Win={"val": 10.0, "units": "lbm/s"},
                    Tout={"val": 14.696, "units": "degR"},
                    Pout={"val": 518.67, "units": "psi"},
                    CS={"val": 10.0, "units": "lbm/s"},
                ),
            )
            self.connect("duct25.Fl_O:stat:W", "hpc_CS.Win")
            self.connect("hpc.Fl_O:tot:T", "hpc_CS.Tout")
            self.connect("hpc.Fl_O:tot:P", "hpc_CS.Pout")

            self.add_subsystem("hpc_EtaBalance", SmallCoreEffBalance(eng_type="large", tech_level=0))
            self.connect("hpc_CS.CS", "hpc_EtaBalance.CS")
            self.connect("hpc.eff_poly", "hpc_EtaBalance.eta_p")
            self.connect("hpc_EtaBalance.eta_a", "hpc.eff")

            self.add_subsystem(
                "fan_dia",
                om.ExecComp(
                    "FanDia = 2.0*(area/(pi*(1.0-hub_tip**2.0)))**0.5",
                    area={"val": 7000.0, "units": "inch**2"},
                    hub_tip={"val": 0.3125, "units": None},
                    FanDia={"val": 100.0, "units": "inch"},
                ),
            )
            self.connect("inlet.Fl_O:stat:area", "fan_dia.area")

            self.add_subsystem(
                "opr_calc",
                om.ExecComp(
                    "OPR_simple = FPR*LPCPR*HPCPR",
                    FPR={"val": 1.3, "units": None},
                    LPCPR={"val": 3.0, "units": None},
                    HPCPR={"val": 14.0, "units": None},
                    OPR_simple={"val": 55.0, "units": None},
                ),
            )

            # order_add = ['hpc_CS', 'fan_dia', 'opr_calc']
            order_add = ["hpc_CS", "hpc_EtaBalance", "fan_dia", "opr_calc"]

        else:

            balance.add_balance("FAR", val=0.017, lower=1e-4, eq_units="lbf")
            self.connect("balance.FAR", "burner.Fl_I:FAR")
            self.connect("perf.Fn", "balance.lhs:FAR")
            # self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            balance.add_balance("W", units="lbm/s", lower=10.0, upper=2500.0, eq_units="inch**2")
            self.connect("balance.W", "fc.W")
            self.connect("core_nozz.Throat:stat:area", "balance.lhs:W")

            balance.add_balance("BPR", lower=15.0, upper=40.0)
            self.connect("balance.BPR", "splitter.BPR")
            self.connect("fan.map.RlineMap", "balance.lhs:BPR")

            balance.add_balance(
                "fan_Nmech", val=2000.0, units="rpm", lower=500.0, eq_units="hp", rhs_val=0.0, res_ref=1e2
            )
            self.connect("balance.fan_Nmech", "Fan_Nmech")
            self.connect("fan_shaft.pwr_net", "balance.lhs:fan_Nmech")

            balance.add_balance(
                "lp_Nmech", val=6000.0, units="rpm", lower=500.0, eq_units="hp", rhs_val=0.0, res_ref=1e2
            )
            self.connect("balance.lp_Nmech", "LP_Nmech")
            self.connect("lp_shaft.pwr_net", "balance.lhs:lp_Nmech")

            balance.add_balance(
                "hp_Nmech", val=20000.0, units="rpm", lower=500.0, eq_units="hp", rhs_val=0.0, res_ref=1e2
            )
            self.connect("balance.hp_Nmech", "HP_Nmech")
            self.connect("hp_shaft.pwr_net", "balance.lhs:hp_Nmech")

            order_add = []

        if cooling:
            self.add_subsystem(
                "hpt_cooling", pyc.TurbineCooling(n_stages=2, thermo_data=self.options["thermo_data"], T_metal=2460.0)
            )
            self.add_subsystem(
                "hpt_chargable", pyc.CombineCooling(n_ins=3)
            )  # Number of cooling flows which are chargable

            self.pyc_connect_flow("bld3.bld_inlet", "hpt_cooling.Fl_cool", connect_stat=False)
            self.pyc_connect_flow("burner.Fl_O", "hpt_cooling.Fl_turb_I")
            self.pyc_connect_flow("hpt.Fl_O", "hpt_cooling.Fl_turb_O")

            self.connect("hpt_cooling.row_1.W_cool", "hpt_chargable.W_1")
            self.connect("hpt_cooling.row_2.W_cool", "hpt_chargable.W_2")
            self.connect("hpt_cooling.row_3.W_cool", "hpt_chargable.W_3")
            self.connect("hpt.power", "hpt_cooling.turb_pwr")

            balance.add_balance("hpt_nochrg_cool_frac", val=0.063660111, lower=0.02, upper=0.15, eq_units="lbm/s")
            self.connect("balance.hpt_nochrg_cool_frac", "bld3.bld_inlet:frac_W")
            self.connect("bld3.bld_inlet:stat:W", "balance.lhs:hpt_nochrg_cool_frac")
            self.connect("hpt_cooling.row_0.W_cool", "balance.rhs:hpt_nochrg_cool_frac")

            balance.add_balance("hpt_chrg_cool_frac", val=0.07037185, lower=0.02, upper=0.15, eq_units="lbm/s")
            self.connect("balance.hpt_chrg_cool_frac", "bld3.bld_exit:frac_W")
            self.connect("bld3.bld_exit:stat:W", "balance.lhs:hpt_chrg_cool_frac")
            self.connect("hpt_chargable.W_cool", "balance.rhs:hpt_chrg_cool_frac")

            order_add = ["hpt_cooling", "hpt_chargable"]

        self.set_order(main_order + order_add + ["balance"])

        # --- Inlet Flow ---
        self.pyc_connect_flow("fc.Fl_O", "inlet.Fl_I")
        self.pyc_connect_flow("inlet.Fl_O", "fan.Fl_I")
        self.pyc_connect_flow("fan.Fl_O", "splitter.Fl_I")
        self.pyc_connect_flow("splitter.Fl_O1", "duct2.Fl_I")
        # ---  Core Flow ---
        self.pyc_connect_flow("duct2.Fl_O", "lpc.Fl_I")
        self.pyc_connect_flow("lpc.Fl_O", "bld25.Fl_I")
        self.pyc_connect_flow("bld25.Fl_O", "duct25.Fl_I")
        self.pyc_connect_flow("duct25.Fl_O", "inject.Fl_I")
        self.pyc_connect_flow("inject.Fl_O", "hpc.Fl_I")  # injector
        self.pyc_connect_flow("hpc.Fl_O", "bld3.Fl_I")
        self.pyc_connect_flow("bld3.Fl_O", "burner.Fl_I")
        self.pyc_connect_flow("burner.Fl_O", "hpt.Fl_I")
        self.pyc_connect_flow("hpt.Fl_O", "duct45.Fl_I")
        self.pyc_connect_flow("duct45.Fl_O", "lpt.Fl_I")

        self.pyc_connect_flow("lpt.Fl_O", "duct5.Fl_I")
        self.pyc_connect_flow("duct5.Fl_O", "extract.Fl_I")  # extractor
        self.pyc_connect_flow("extract.Fl_O", "core_nozz.Fl_I")

        # --- Bypass Flow ---
        self.pyc_connect_flow("splitter.Fl_O2", "byp_bld.Fl_I")
        self.pyc_connect_flow("byp_bld.Fl_O", "duct17.Fl_I")
        self.pyc_connect_flow("duct17.Fl_O", "byp_nozz.Fl_I")

        # Connect Bleed Flows
        self.pyc_connect_flow("hpc.bld_inlet", "lpt.bld_inlet", connect_stat=False)
        self.pyc_connect_flow("hpc.bld_exit", "lpt.bld_exit", connect_stat=False)
        self.pyc_connect_flow("bld3.bld_inlet", "hpt.bld_inlet", connect_stat=False)
        self.pyc_connect_flow("bld3.bld_exit", "hpt.bld_exit", connect_stat=False)

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options["atol"] = 1e-4
        newton.options["rtol"] = 1e-4
        newton.options["iprint"] = 2
        newton.options["maxiter"] = 10
        newton.options["solve_subsystems"] = True
        newton.options["max_sub_solves"] = 10
        newton.options["reraise_child_analysiserror"] = False
        # newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch = om.ArmijoGoldsteinLS()
        newton.linesearch.options["rho"] = 0.75
        # newton.linesearch.options["maxiter"] = 1
        newton.linesearch.options["iprint"] = -1

        self.linear_solver = om.DirectSolver()

        super().setup()


def viewer(prob, pt, file=sys.stdout):
    """
    print a report of all the relevant cycle properties
    """

    summary_data = (
        prob[pt + ".fc.Fl_O:stat:MN"],
        prob[pt + ".fc.alt"],
        prob[pt + ".inlet.Fl_O:stat:W"],
        prob[pt + ".perf.Fn"],
        prob[pt + ".perf.Fg"],
        prob[pt + ".inlet.F_ram"],
        prob[pt + ".perf.OPR"],
        prob[pt + ".perf.TSFC"],
        prob[pt + ".tsec_perf.TSEC"],
        prob[pt + ".splitter.BPR"],
        prob[pt + ".extract.sub_flow.W_water"],
        prob[pt + ".inject.mix:W"],
    )

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print(
        "    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC      TSEC      BPR      Wext      Winj ",
        file=file,
        flush=True,
    )
    print(
        " %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f  %7.5f  %7.3f  %7.5f  %7.5f" % summary_data,  # %7.5f
        file=file,
        flush=True,
    )

    fs_names = [
        "fc.Fl_O",
        "inlet.Fl_O",
        "fan.Fl_O",
        "splitter.Fl_O1",
        "duct2.Fl_O",
        "lpc.Fl_O",
        "bld25.Fl_O",
        "duct25.Fl_O",
        "inject.Fl_O",
        "hpc.Fl_O",
        "bld3.Fl_O",
        "burner.Fl_O",
        "hpt.Fl_O",
        "duct45.Fl_O",
        "lpt.Fl_O",
        "extract.Fl_O",
        "duct5.Fl_O",
        "core_nozz.Fl_O",
        "splitter.Fl_O2",
        "byp_bld.Fl_O",
        "duct17.Fl_O",
        "byp_nozz.Fl_O",
    ]
    fs_full_names = [f"{pt}.{fs}" for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    comp_names = ["fan", "lpc", "hpc"]
    comp_full_names = [f"{pt}.{c}" for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    pyc.print_burner(prob, [f"{pt}.burner"], file=file)

    turb_names = ["hpt", "lpt"]
    turb_full_names = [f"{pt}.{t}" for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    noz_names = ["core_nozz", "byp_nozz"]
    noz_full_names = [f"{pt}.{n}" for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

    shaft_names = ["hp_shaft", "lp_shaft", "fan_shaft"]
    shaft_full_names = [f"{pt}.{s}" for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)

    bleed_names = ["hpc", "bld3", "bld3", "bld25"]
    bleed_full_names = [f"{pt}.{b}" for b in bleed_names]
    pyc.print_bleed(prob, bleed_full_names, file=file)


class MPN3(pyc.MPCycle):
    def initialize(self):
        self.options.declare("order_add", default=[], desc="Name of subsystems to add to end of order.")
        self.options.declare("order_start", default=[], desc="Name of subsystems to add to beginning of order.")
        self.options.declare("statics", default=True, desc="Tells the model whether or not to connect areas.")
        self.options.declare("use_h2", default=False, desc="If True, tells the model to use hydrogen fuel.")
        self.options.declare("wet_air", default=False, desc="If True, use wet air.")
        self.options.declare("design_water", default=False, types=bool, desc="If True, set DP as water injection.")

        super().initialize()

    def setup(self):
        use_h2 = self.options["use_h2"]
        wet_air = self.options["wet_air"]

        # alt_war = 0.00001  # water-air ratio of atmosphere
        # sls_war = 0.00001  # water-air ratio of atmosphere
        alt_war = 0.001  # water-air ratio of atmosphere
        sls_war = 0.007  # water-air ratio of atmosphere

        # TOC POINT (DESIGN)
        self.pyc_add_pnt(
            "TOC",
            N3(use_h2=use_h2, wet_air=wet_air),
            promotes_inputs=[
                ("fan.PR", "fan:PRdes"),
                ("lpc.PR", "lpc:PRdes"),
                ("opr_calc.FPR", "fan:PRdes"),
                ("opr_calc.LPCPR", "lpc:PRdes"),
            ],
        )

        # POINT 1: Top-of-climb (TOC)
        self.set_input_defaults("TOC.fc.alt", 35000.0, units="ft"),
        self.set_input_defaults("TOC.fc.MN", 0.8),
        self.set_input_defaults("TOC.inlet.ram_recovery", 0.9980),

        self.set_input_defaults("TOC.balance.rhs:fan_eff", 0.97)
        self.set_input_defaults("TOC.duct2.dPqP", 0.0100)
        self.set_input_defaults("TOC.balance.rhs:lpc_eff", 0.905),
        self.set_input_defaults("TOC.duct25.dPqP", 0.0150),
        self.set_input_defaults("TOC.balance.rhs:hpt_eff", 0.91),
        self.set_input_defaults("TOC.duct45.dPqP", 0.0050),
        self.set_input_defaults("TOC.balance.rhs:lpt_eff", 0.92),
        self.set_input_defaults("TOC.duct5.dPqP", 0.0100),
        self.set_input_defaults("TOC.duct17.dPqP", 0.0150),
        self.set_input_defaults("TOC.Fan_Nmech", 2184.5, units="rpm"),
        self.set_input_defaults("TOC.LP_Nmech", 6772.0, units="rpm"),
        self.set_input_defaults("TOC.HP_Nmech", 20871.0, units="rpm"),

        self.set_input_defaults("TOC.inlet.MN", 0.625),
        self.set_input_defaults("TOC.fan.MN", 0.45)
        self.set_input_defaults("TOC.splitter.MN1", 0.45)
        self.set_input_defaults("TOC.splitter.MN2", 0.45)
        self.set_input_defaults("TOC.duct2.MN", 0.45),
        self.set_input_defaults("TOC.lpc.MN", 0.45),
        self.set_input_defaults("TOC.bld25.MN", 0.45),
        self.set_input_defaults("TOC.duct25.MN", 0.45),
        self.set_input_defaults("TOC.hpc.MN", 0.30),
        self.set_input_defaults("TOC.bld3.MN", 0.30)
        self.set_input_defaults("TOC.burner.MN", 0.10),
        self.set_input_defaults("TOC.hpt.MN", 0.30),
        self.set_input_defaults("TOC.duct45.MN", 0.45),
        self.set_input_defaults("TOC.lpt.MN", 0.35),
        self.set_input_defaults("TOC.duct5.MN", 0.25),
        self.set_input_defaults("TOC.byp_bld.MN", 0.45),
        self.set_input_defaults("TOC.duct17.MN", 0.45),
        self.set_input_defaults("TOC.fc.WAR", alt_war),
        self.set_input_defaults("TOC.inject.MN", 0.45),
        self.set_input_defaults("TOC.extract.MN", 0.25),
        self.set_input_defaults("TOC.extract.sub_flow.w_frac", 0.05),

        self.pyc_add_cycle_param("burner.dPqP", 0.0400),
        self.pyc_add_cycle_param("core_nozz.Cv", 0.9999),
        self.pyc_add_cycle_param("byp_nozz.Cv", 0.9975),
        self.pyc_add_cycle_param("lp_shaft.fracLoss", 0.01)
        self.pyc_add_cycle_param("hp_shaft.HPX", 350.0, units="hp"),
        self.pyc_add_cycle_param("bld25.sbv:frac_W", 0.0),
        self.pyc_add_cycle_param("hpc.bld_inlet:frac_W", 0.0),
        self.pyc_add_cycle_param("hpc.bld_inlet:frac_P", 0.1465),
        self.pyc_add_cycle_param("hpc.bld_inlet:frac_work", 0.5),
        self.pyc_add_cycle_param("hpc.bld_exit:frac_W", 0.02),
        self.pyc_add_cycle_param("hpc.bld_exit:frac_P", 0.1465),
        self.pyc_add_cycle_param("hpc.bld_exit:frac_work", 0.5),
        self.pyc_add_cycle_param("hpc.cust:frac_W", 0.0),
        self.pyc_add_cycle_param("hpc.cust:frac_P", 0.1465),
        self.pyc_add_cycle_param("hpc.cust:frac_work", 0.35),
        self.pyc_add_cycle_param("hpt.bld_inlet:frac_P", 1.0),
        self.pyc_add_cycle_param("hpt.bld_exit:frac_P", 0.0),
        self.pyc_add_cycle_param("lpt.bld_inlet:frac_P", 1.0),
        self.pyc_add_cycle_param("lpt.bld_exit:frac_P", 0.0),
        self.pyc_add_cycle_param("byp_bld.bypBld:frac_W", 0.0),
        self.pyc_add_cycle_param("inject.dPqP", 0.0),
        self.pyc_add_cycle_param("extract.dPqP", 0.0),

        # OTHER POINTS (OFF-DESIGN)
        self.od_pts = ["RTO"]
        self.cooling = [True]
        self.od_MNs = [0.25]
        self.od_alts = [0.0]
        self.od_dTs = [27.0]
        self.od_BPRs = [1.75]
        self.od_recoveries = [0.9970]
        self.war = [sls_war]
        self.w_frac = [0.0]

        for i, pt in enumerate(self.od_pts):
            self.pyc_add_pnt(pt, N3(design=False, use_h2=use_h2, wet_air=wet_air, cooling=self.cooling[i]))

            self.set_input_defaults(pt + ".fc.MN", val=self.od_MNs[i])
            self.set_input_defaults(pt + ".fc.alt", val=self.od_alts[i], units="ft")
            self.set_input_defaults(pt + ".fc.dTs", val=self.od_dTs[i], units="degR")
            self.set_input_defaults(pt + ".balance.rhs:BPR", val=self.od_BPRs[i])
            self.set_input_defaults(pt + ".inlet.ram_recovery", val=self.od_recoveries[i])
            self.set_input_defaults(pt + ".fc.WAR", val=self.war[i])
            self.set_input_defaults(pt + ".extract.sub_flow.w_frac", self.w_frac[i]),

        # Extra set input for Rolling Takeoff
        self.set_input_defaults("RTO.balance.rhs:FAR", 22800.0, units="lbf"),

        self.pyc_connect_des_od("fan.s_PR", "fan.s_PR")
        self.pyc_connect_des_od("fan.s_Wc", "fan.s_Wc")
        self.pyc_connect_des_od("fan.s_eff", "fan.s_eff")
        self.pyc_connect_des_od("fan.s_Nc", "fan.s_Nc")
        self.pyc_connect_des_od("lpc.s_PR", "lpc.s_PR")
        self.pyc_connect_des_od("lpc.s_Wc", "lpc.s_Wc")
        self.pyc_connect_des_od("lpc.s_eff", "lpc.s_eff")
        self.pyc_connect_des_od("lpc.s_Nc", "lpc.s_Nc")
        self.pyc_connect_des_od("hpc.s_PR", "hpc.s_PR")
        self.pyc_connect_des_od("hpc.s_Wc", "hpc.s_Wc")
        self.pyc_connect_des_od("hpc.s_eff", "hpc.s_eff")
        self.pyc_connect_des_od("hpc.s_Nc", "hpc.s_Nc")
        self.pyc_connect_des_od("hpt.s_PR", "hpt.s_PR")
        self.pyc_connect_des_od("hpt.s_Wp", "hpt.s_Wp")
        self.pyc_connect_des_od("hpt.s_eff", "hpt.s_eff")
        self.pyc_connect_des_od("hpt.s_Np", "hpt.s_Np")
        self.pyc_connect_des_od("lpt.s_PR", "lpt.s_PR")
        self.pyc_connect_des_od("lpt.s_Wp", "lpt.s_Wp")
        self.pyc_connect_des_od("lpt.s_eff", "lpt.s_eff")
        self.pyc_connect_des_od("lpt.s_Np", "lpt.s_Np")

        self.pyc_connect_des_od("gearbox.gear_ratio", "gearbox.gear_ratio")
        self.pyc_connect_des_od("core_nozz.Throat:stat:area", "balance.rhs:W")

        if self.options["statics"] is True:
            self.pyc_connect_des_od("inlet.Fl_O:stat:area", "inlet.area")
            self.pyc_connect_des_od("fan.Fl_O:stat:area", "fan.area")
            self.pyc_connect_des_od("splitter.Fl_O1:stat:area", "splitter.area1")
            self.pyc_connect_des_od("splitter.Fl_O2:stat:area", "splitter.area2")
            self.pyc_connect_des_od("duct2.Fl_O:stat:area", "duct2.area")
            self.pyc_connect_des_od("lpc.Fl_O:stat:area", "lpc.area")
            self.pyc_connect_des_od("bld25.Fl_O:stat:area", "bld25.area")
            self.pyc_connect_des_od("duct25.Fl_O:stat:area", "duct25.area")
            self.pyc_connect_des_od("hpc.Fl_O:stat:area", "hpc.area")
            self.pyc_connect_des_od("bld3.Fl_O:stat:area", "bld3.area")
            self.pyc_connect_des_od("burner.Fl_O:stat:area", "burner.area")
            self.pyc_connect_des_od("hpt.Fl_O:stat:area", "hpt.area")
            self.pyc_connect_des_od("duct45.Fl_O:stat:area", "duct45.area")
            self.pyc_connect_des_od("lpt.Fl_O:stat:area", "lpt.area")
            self.pyc_connect_des_od("duct5.Fl_O:stat:area", "duct5.area")
            self.pyc_connect_des_od("byp_bld.Fl_O:stat:area", "byp_bld.area")
            self.pyc_connect_des_od("duct17.Fl_O:stat:area", "duct17.area")
            self.pyc_connect_des_od("extract.Fl_O:stat:area", "extract.area")
            self.pyc_connect_des_od("inject.Fl_O:stat:area", "inject.area")

        self.pyc_connect_des_od("duct2.s_dPqP", "duct2.s_dPqP")
        self.pyc_connect_des_od("duct25.s_dPqP", "duct25.s_dPqP")
        self.pyc_connect_des_od("duct45.s_dPqP", "duct45.s_dPqP")
        self.pyc_connect_des_od("duct5.s_dPqP", "duct5.s_dPqP")
        self.pyc_connect_des_od("duct17.s_dPqP", "duct17.s_dPqP")

        self.connect("RTO.balance.hpt_chrg_cool_frac", "TOC.bld3.bld_exit:frac_W")
        self.connect("RTO.balance.hpt_nochrg_cool_frac", "TOC.bld3.bld_inlet:frac_W")

        self.add_subsystem(
            "T4_ratio",
            om.ExecComp(
                "TOC_T4 = RTO_T4*TR",
                RTO_T4={"val": 3400.0, "units": "degR"},
                TOC_T4={"val": 3150.0, "units": "degR"},
                TR={"val": 0.926470588, "units": None},
            ),
            promotes_inputs=["RTO_T4"],
        )
        self.connect("T4_ratio.TOC_T4", "TOC.balance.rhs:FAR")

        self.connect("TOC.extract.W_water", "TOC.inject.mix:W")
        self.connect("RTO.extract.W_water", "RTO.inject.mix:W")

        initial_order = ["T4_ratio", "TOC", "RTO"]
        self.set_order(self.options["order_start"] + initial_order + self.options["order_add"])

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options["atol"] = 1e-6
        newton.options["rtol"] = 1e-6
        newton.options["iprint"] = 2
        newton.options["maxiter"] = 10
        newton.options["solve_subsystems"] = True
        newton.options["max_sub_solves"] = 10
        newton.options["err_on_non_converge"] = False
        newton.options["reraise_child_analysiserror"] = False
        newton.options["restart_from_successful"] = True
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options["bound_enforcement"] = "scalar"
        newton.linesearch.options["iprint"] = -1

        self.linear_solver = om.DirectSolver(assemble_jac=True)

        super().setup()


def N3ref_model(use_h2=False, wet_air=True):

    prob = om.Problem()

    prob.model = MPN3(use_h2=use_h2, wet_air=wet_air)

    # setup the optimization
    prob.driver = om.ScipyOptimizeDriver()
    prob.driver.options["optimizer"] = "SLSQP"
    prob.driver.options["debug_print"] = ["desvars", "nl_cons", "objs"]
    prob.driver.opt_settings = {"Major step limit": 0.05}

    prob.model.add_design_var("fan:PRdes", lower=1.20, upper=1.4)
    prob.model.add_design_var("lpc:PRdes", lower=2.0, upper=4.0)
    prob.model.add_design_var("TOC.balance.rhs:hpc_PR", lower=40.0, upper=70.0, ref0=40.0, ref=70.0)
    prob.model.add_design_var("RTO_T4", lower=3000.0, upper=3600.0, ref0=3000.0, ref=3600.0)
    prob.model.add_design_var("T4_ratio.TR", lower=0.5, upper=0.95, ref0=0.5, ref=0.95)

    prob.model.add_objective("TOC.perf.TSFC")

    # to add the constraint to the model
    prob.model.add_constraint("TOC.fan_dia.FanDia", upper=100.0, ref=100.0)

    return prob


def map_plots(prob, pt):
    comp_names = ["fan", "lpc", "hpc"]
    comp_full_names = [f"{pt}.{c}" for c in comp_names]
    pyc.plot_compressor_maps(prob, comp_full_names)

    turb_names = ["lpt", "hpt"]
    turb_full_names = [f"{pt}.{c}" for c in turb_names]
    pyc.plot_turbine_maps(prob, turb_full_names)


if __name__ == "__main__":

    import time

    # prob = N3ref_model(use_h2=False)
    prob = N3ref_model(use_h2=True)

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
    prob.set_val("RTO.hpt_cooling.x_factor", 0.9)

    # Set initial guesses for balances
    if prob.model.options["use_h2"]:
        prob["TOC.balance.FAR"] = 0.0102
        prob["TOC.balance.lpt_PR"] = 9.8
        prob["TOC.balance.hpt_PR"] = 4.2
        prob["TOC.fc.balance.Pt"] = 5.2
        prob["TOC.fc.balance.Tt"] = 444.3
        prob["TOC.inject.mix:W"] = 0.001

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
        prob["TOC.inject.mix:W"] = 0.001

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

    run_sweep = True
    save_res = True
    # run_var = "HPC"
    # run_var = "FAN"
    # run_var = "LPC"
    # run_var = "T4Ratio"
    run_var = "T4"
    # run_var = "TOC_wfrac"
    # run_var = "TOC_BPR"

    if run_sweep:
        n = 10
        if run_var == "HPC":
            # var = np.linspace(40, 60, n)  # JetA
            var = np.linspace(40, 61, n)  # H2
            var_namepath = "TOC.balance.rhs:hpc_PR"
        elif run_var == "FAN":
            # var = np.linspace(1.14, 1.33, n)  # JetA
            var = np.linspace(1.15, 1.33, n)  # H2
            var_namepath = "fan:PRdes"
        elif run_var == "LPC":
            # var = np.linspace(1.18, 11.0, n)  # JetA
            var = np.linspace(2.0, 18, n)  # H2 1.31
            var_namepath = "lpc:PRdes"
        elif run_var == "T4Ratio":
            # var = np.linspace(0.89, 0.99, n)  # JetA
            var = np.linspace(0.89, 0.99, n)  # H2
            var_namepath = "T4_ratio.TR"
        elif run_var == "T4":
            # var = np.linspace(3270, 3600, n)  # JetA
            var = np.linspace(3270, 3600, n)  # H2
            var_namepath = "RTO_T4"
        elif run_var == "TOC_wfrac":
            # var = np.linspace(0.0, 0.1, n)  # JetA
            var = np.linspace(0.0, 0.07, n)  # H2
            var_namepath = "TOC.extract.sub_flow.w_frac"
        elif run_var == "TOC_BPR":
            # var = np.linspace(20, 30, n)  # JetA
            var = np.linspace(20, 30, n)  # H2
            var_namepath = "TOC.splitter.BPR"

        TSFC_TOC = np.zeros(var.size)
        TSFC_RTO = np.zeros(var.size)
        TSEC_TOC = np.zeros(var.size)
        TSEC_RTO = np.zeros(var.size)
        TOC_thrust = np.zeros(var.size)
        Wwater = np.zeros(var.size)

        print("### Running " + var_namepath + " Initialization ###")
        prob.run_model()

        for i, vari in enumerate(var):
            print("### Running " + var_namepath + f"={vari:.2f}, Iteration: {i+1} ###")

            prob[var_namepath] = vari
            prob.run_model()

            TSFC_TOC[i] = prob.get_val("TOC.perf.TSFC")
            TSFC_RTO[i] = prob.get_val("RTO.perf.TSFC")
            TSEC_TOC[i] = prob.get_val("TOC.tsec_perf.TSEC")
            TSEC_RTO[i] = prob.get_val("RTO.tsec_perf.TSEC")
            TOC_thrust[i] = prob.get_val("TOC.perf.Fn")
            Wwater[i] = prob.get_val("TOC.inject.mix:W")

            # if prob.model.options["use_h2"]:
            #     fname = "../OUTPUT/N3_trends/bound_sweeps/H2/N3_" + var_namepath + f"_{vari:.2f}.pkl"
            # else:
            #     fname = "../OUTPUT/N3_trends/bound_sweeps/JetA/N3_" + var_namepath + f"_{vari:.2f}.pkl"

            # with open(fname, "wb") as f:
            #     pkl.dump(
            #         np.vstack((vari, Wwater[i], TSFC_TOC[i], TSFC_RTO[i], TSEC_TOC[i], TSEC_RTO[i], TOC_thrust[i])), f
            #     )

        if prob.model.options["use_h2"]:
            fname = "../OUTPUT/N3_trends/bound_sweeps/H2/N3_" + var_namepath + ".pkl"
        else:
            fname = "../OUTPUT/N3_trends/bound_sweeps/JetA/N3_" + var_namepath + ".pkl"

        with open(fname, "wb") as f:
            pkl.dump(np.vstack((var, Wwater, TSFC_TOC, TSFC_RTO, TSEC_TOC, TSEC_RTO, TOC_thrust)), f)

    else:
        wfrac = 0.0000
        prob.set_val("TOC.extract.sub_flow.w_frac", wfrac)
        prob.set_val("RTO.extract.sub_flow.w_frac", wfrac)

        # prob.model.RTO.nonlinear_solver.options["maxiter"] = 2
        # prob.model.SLS.nonlinear_solver.options["maxiter"] = 0
        # prob.model.CRZ.nonlinear_solver.options["maxiter"] = 0
        # prob.model.nonlinear_solver.options["maxiter"] = 0
        # prob.model.nonlinear_solver.options["err_on_non_converge"] = False

        prob.run_model()
        # prob.model.list_outputs(residuals=True, explicit=True, includes="RTO*", residuals_tol=1e-2, prom_name=True)
        # prob.check_partials(compact_print=True, show_only_incorrect=True, method="fd")

        if save_res is True:
            output_dir = f"../OUTPUT/N3_output/CLVR_JetA_{vari}"
            if os.path.isdir(output_dir) is False:
                os.mkdir(output_dir)
            with open(f"{output_dir}/output.txt", "w") as file:
                for pt in ["TOC", "RTO"]:
                    viewer(prob, pt, file)

    print("time", time.time() - st)
