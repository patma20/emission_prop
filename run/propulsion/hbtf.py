# --- Python 3.8 ---
"""
High bypass turbofan model in pyCycle
"""
# ==============================================================================
# Standard Python modules
# ==============================================================================
import sys

# ==============================================================================
# External Python modules
# ==============================================================================
import openmdao.api as om
import pycycle.api as pyc

# from openconcept.components.heat_exchanger import HXGroup as HeatExchanger
from openconcept.thermal import LiquidCooledComp, CoolantReservoir
from openmdao.api import IndepVarComp

# ==============================================================================
# Extension modules
# ==============================================================================
from components.misc_components import byp_pressure, HeatExchanger, GetFluidProps
import constants.constants as con
from components.constraint_balance import FAR_Balance_2


class HBTF(pyc.Cycle):
    def initialize(self):
        self.options.declare(
            "constrained_balance", default=True, types=bool, desc="Flag for balance type in off-design"
        )
        super().initialize()

    def setup(self):
        # Setup the problem by including all the relavant components here - comp, burner, turbine etc
        # Create any relavent short hands here:
        design = self.options["design"]
        con_bal_flag = self.options["constrained_balance"]

        USE_TABULAR = True
        if USE_TABULAR:
            self.options["thermo_method"] = "TABULAR"
            self.options["thermo_data"] = pyc.AIR_JETA_TAB_SPEC
            FUEL_TYPE = "FAR"
        else:
            self.options["thermo_method"] = "CEA"
            self.options["thermo_data"] = pyc.species_data.janaf
            FUEL_TYPE = "Jet-A(g)"
        self.add_subsystem("fc", pyc.FlightConditions())
        self.add_subsystem("inlet", pyc.Inlet())
        self.add_subsystem(
            "fan",
            pyc.Compressor(map_data=pyc.FanMap, bleed_names=[], map_extrap=True),
            promotes_inputs=[("Nmech", "LP_Nmech")],
        )
        self.add_subsystem("splitter", pyc.Splitter())
        self.add_subsystem("duct4", pyc.Duct())
        self.add_subsystem(
            "lpc", pyc.Compressor(map_data=pyc.LPCMap, map_extrap=True), promotes_inputs=[("Nmech", "LP_Nmech")]
        )
        self.add_subsystem("duct6", pyc.Duct())
        self.add_subsystem(
            "hpc",
            pyc.Compressor(map_data=pyc.HPCMap, bleed_names=["cool1", "cool2", "cust"], map_extrap=True),
            promotes_inputs=[("Nmech", "HP_Nmech")],
        )
        self.add_subsystem("bld3", pyc.BleedOut(bleed_names=["cool3", "cool4"]))
        self.add_subsystem("burner", pyc.Combustor(fuel_type=FUEL_TYPE))
        self.add_subsystem(
            "hpt",
            pyc.Turbine(map_data=pyc.HPTMap, bleed_names=["cool3", "cool4"], map_extrap=True),
            promotes_inputs=[("Nmech", "HP_Nmech")],
        )
        self.add_subsystem("duct11", pyc.Duct())
        self.add_subsystem(
            "lpt",
            pyc.Turbine(map_data=pyc.LPTMap, bleed_names=["cool1", "cool2"], map_extrap=True),
            promotes_inputs=[("Nmech", "LP_Nmech")],
        )
        self.add_subsystem("duct13", pyc.Duct())
        self.add_subsystem("core_nozz", pyc.Nozzle(nozzType="CV", lossCoef="Cv"))
        self.add_subsystem("byp_bld", pyc.BleedOut(bleed_names=["bypBld"]))
        self.add_subsystem("duct15", pyc.Duct())
        self.add_subsystem("hx", HeatExchanger(num_nodes=1))

        thermal_params = self.add_subsystem("thermal_params", IndepVarComp(), promotes_outputs=["*"])
        thermal_params.add_output("mdot_coolant", val=0.2, units="kg/s")
        thermal_params.add_output("rho_coolant", val=con.rho_oil, units="kg/m**3")
        thermal_params.add_output("heat_load", 50.0, units="W")
        thermal_params.add_output("coolant_mass", val=10.0, units="kg")
        thermal_params.add_output("channel_width", val=1, units="mm")
        thermal_params.add_output("channel_height", val=20, units="mm")
        thermal_params.add_output("channel_length", val=0.2, units="m")
        thermal_params.add_output("n_parallel", val=50)

        self.add_subsystem(
            "heatsink",
            LiquidCooledComp(num_nodes=1, specific_heat_coolant=2500, quasi_steady=True),
            promotes_inputs=["channel_*", "n_parallel"],
        )  # upper value of cp of engine oil
        self.add_subsystem("reservoir", CoolantReservoir(num_nodes=1), promotes_inputs=[("mass", "coolant_mass")])
        self.add_subsystem("byp_P", byp_pressure())
        self.add_subsystem("byp_nozz", pyc.Nozzle(nozzType="CV", lossCoef="Cv"))

        # Create shaft instances. Note that LP shaft has 3 ports! => no gearbox
        self.add_subsystem("lp_shaft", pyc.Shaft(num_ports=3), promotes_inputs=[("Nmech", "LP_Nmech")])
        self.add_subsystem("hp_shaft", pyc.Shaft(num_ports=2), promotes_inputs=[("Nmech", "HP_Nmech")])
        self.add_subsystem("perf", pyc.Performance(num_nozzles=2, num_burners=1))

        # Add fluid properties for air to get k and mu
        self.add_subsystem("air", GetFluidProps(fluid_species="air"))

        # Connect the inputs to perf group
        self.connect("inlet.Fl_O:tot:P", "perf.Pt2")
        self.connect("hpc.Fl_O:tot:P", "perf.Pt3")
        self.connect("burner.Wfuel", "perf.Wfuel_0")
        self.connect("inlet.F_ram", "perf.ram_drag")
        self.connect("core_nozz.Fg", "perf.Fg_0")
        self.connect("byp_nozz.Fg", "perf.Fg_1")
        # LP-shaft connections
        self.connect("fan.trq", "lp_shaft.trq_0")
        self.connect("lpc.trq", "lp_shaft.trq_1")
        self.connect("lpt.trq", "lp_shaft.trq_2")
        # HP-shaft connections
        self.connect("hpc.trq", "hp_shaft.trq_0")
        self.connect("hpt.trq", "hp_shaft.trq_1")
        # Ideally expanding flow by conneting flight condition static pressure to nozzle exhaust pressure
        self.connect("fc.Fl_O:stat:P", "core_nozz.Ps_exhaust")
        self.connect("fc.Fl_O:stat:P", "byp_nozz.Ps_exhaust")
        # --- Heat exchanger loop connections ---
        # Mass flow connections
        self.connect("duct15.Fl_O:stat:W", "hx.mdot_cold")
        self.connect("mdot_coolant", ["heatsink.mdot_coolant", "hx.mdot_hot", "reservoir.mdot_coolant"])
        # Temperature connections
        self.connect("duct15.Fl_O:stat:T", "hx.T_in_cold")
        self.connect("hx.T_out_hot", "reservoir.T_in")
        self.connect("heatsink.T_out", "hx.T_in_hot")
        self.connect("reservoir.T_out", "heatsink.T_in")
        # Pressure connetions
        self.connect("byp_P.dPqP", "duct15.dPqP")
        self.connect("duct15.Fl_O:tot:P", "byp_P.Pt_in")
        self.connect("hx.delta_p_cold", "byp_P.delta_p")
        # Thermal property connections
        self.connect("duct15.Fl_O:stat:rho", "hx.rho_cold")
        self.connect("rho_coolant", "hx.rho_hot")
        self.connect("duct15.Fl_O:stat:Cp", "hx.cp_cold")
        self.connect("duct15.Fl_O:stat:T", "air.T")
        self.connect("duct15.Fl_O:stat:P", "air.P")
        self.connect("air.mu", "hx.mu_cold")
        self.connect("air.k", "hx.k_cold")
        # Heat flow connections
        self.connect("hx.heat_transfer", "duct15.Q_dot")
        self.connect("heat_load", "heatsink.q_in")

        # Create a balance component
        # Balances can be a bit confusing, here's some explanation -
        #   State Variables:
        #           (W)        Inlet mass flow rate to implictly balance thrust
        #                      LHS: perf.Fn  == RHS: Thrust requirement (set when TF is instantiated)
        #
        #           (FAR)      Fuel-air ratio to balance Tt4
        #                      LHS: burner.Fl_O:tot:T  == RHS: Tt4 target (set when TF is instantiated)
        #
        #           (lpt_PR)   LPT press ratio to balance shaft power on the low spool
        #           (hpt_PR)   HPT press ratio to balance shaft power on the high spool
        balance = self.add_subsystem("balance", om.BalanceComp())
        if design:
            balance.add_balance("FAR", eq_units="degR", lower=1e-4, val=0.017)
            self.connect("balance.FAR", "burner.Fl_I:FAR")
            self.connect("burner.Fl_O:tot:T", "balance.lhs:FAR")
            # Note that for the following two balances the mult val is set to -1 so that the NET torque is zero
            balance.add_balance("lpt_PR", val=1.5, lower=1.001, upper=8, eq_units="hp", use_mult=True, mult_val=-1)
            self.connect("balance.lpt_PR", "lpt.PR")
            self.connect("lp_shaft.pwr_in_real", "balance.lhs:lpt_PR")
            self.connect("lp_shaft.pwr_out_real", "balance.rhs:lpt_PR")
            balance.add_balance("hpt_PR", val=1.5, lower=1.001, upper=8, eq_units="hp", use_mult=True, mult_val=-1)
            self.connect("balance.hpt_PR", "hpt.PR")
            self.connect("hp_shaft.pwr_in_real", "balance.lhs:hpt_PR")
            self.connect("hp_shaft.pwr_out_real", "balance.rhs:hpt_PR")
        else:
            # In OFF-DESIGN mode we need to redefine the balances:
            #   State Variables:
            #           (W)        Inlet mass flow rate to balance core flow area
            #                      LHS: core_nozz.Throat:stat:area == Area from DESIGN calculation
            #
            #           (FAR)      Fuel-air ratio to balance Thrust req.
            #                      LHS: perf.Fn  == RHS: Thrust requirement (set when TF is instantiated)
            #
            #           (BPR)      Bypass ratio to balance byp. noz. area
            #                      LHS: byp_nozz.Throat:stat:area == Area from DESIGN calculation
            #
            #           (lp_Nmech)   LP spool speed to balance shaft power on the low spool
            #           (hp_Nmech)   HP spool speed to balance shaft power on the high spool

            if not con_bal_flag:
                balance.add_balance("FAR", val=0.017, lower=1e-4, eq_units="lbf")
                self.connect("balance.FAR", "burner.Fl_I:FAR")
                self.connect("perf.Fn", "balance.lhs:FAR")
            else:
                self.add_subsystem("FAR", FAR_Balance_2())
                self.connect("FAR.FAR", "burner.Fl_I:FAR")
                self.connect("burner.Fl_O:tot:T", "FAR.T4")
                self.connect("bld3.Fl_O:tot:T", "FAR.T3")
                self.connect("fan.map.NcMap", "FAR.NcMapVal")

            balance.add_balance("W", units="lbm/s", lower=10.0, upper=1000.0, eq_units="inch**2")
            self.connect("balance.W", "fc.W")
            self.connect("core_nozz.Throat:stat:area", "balance.lhs:W")

            balance.add_balance("BPR", lower=2.0, upper=10.0, eq_units="inch**2")
            self.connect("balance.BPR", "splitter.BPR")
            self.connect("byp_nozz.Throat:stat:area", "balance.lhs:BPR")
            # Again for the following two balances the mult val is set to -1 so that the NET torque is zero
            balance.add_balance(
                "lp_Nmech", val=1.5, units="rpm", lower=500.0, eq_units="hp", use_mult=True, mult_val=-1
            )
            self.connect("balance.lp_Nmech", "LP_Nmech")
            self.connect("lp_shaft.pwr_in_real", "balance.lhs:lp_Nmech")
            self.connect("lp_shaft.pwr_out_real", "balance.rhs:lp_Nmech")

            balance.add_balance(
                "hp_Nmech", val=1.5, units="rpm", lower=500.0, eq_units="hp", use_mult=True, mult_val=-1
            )
            self.connect("balance.hp_Nmech", "HP_Nmech")
            self.connect("hp_shaft.pwr_in_real", "balance.lhs:hp_Nmech")
            self.connect("hp_shaft.pwr_out_real", "balance.rhs:hp_Nmech")
        # Set up all the flow connections:
        self.pyc_connect_flow("fc.Fl_O", "inlet.Fl_I")
        self.pyc_connect_flow("inlet.Fl_O", "fan.Fl_I")
        self.pyc_connect_flow("fan.Fl_O", "splitter.Fl_I")
        self.pyc_connect_flow("splitter.Fl_O1", "duct4.Fl_I")
        self.pyc_connect_flow("duct4.Fl_O", "lpc.Fl_I")
        self.pyc_connect_flow("lpc.Fl_O", "duct6.Fl_I")
        self.pyc_connect_flow("duct6.Fl_O", "hpc.Fl_I")
        self.pyc_connect_flow("hpc.Fl_O", "bld3.Fl_I")
        self.pyc_connect_flow("bld3.Fl_O", "burner.Fl_I")
        self.pyc_connect_flow("burner.Fl_O", "hpt.Fl_I")
        self.pyc_connect_flow("hpt.Fl_O", "duct11.Fl_I")
        self.pyc_connect_flow("duct11.Fl_O", "lpt.Fl_I")
        self.pyc_connect_flow("lpt.Fl_O", "duct13.Fl_I")
        self.pyc_connect_flow("duct13.Fl_O", "core_nozz.Fl_I")
        self.pyc_connect_flow("splitter.Fl_O2", "byp_bld.Fl_I")
        self.pyc_connect_flow("byp_bld.Fl_O", "duct15.Fl_I")
        self.pyc_connect_flow("duct15.Fl_O", "byp_nozz.Fl_I")
        # Bleed flows:
        self.pyc_connect_flow("hpc.cool1", "lpt.cool1", connect_stat=False)
        self.pyc_connect_flow("hpc.cool2", "lpt.cool2", connect_stat=False)
        self.pyc_connect_flow("bld3.cool3", "hpt.cool3", connect_stat=False)
        self.pyc_connect_flow("bld3.cool4", "hpt.cool4", connect_stat=False)

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

        # Specify solver settings:
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options["atol"] = 1e-7
        # set this very small, so it never activates and we rely on atol
        newton.options["rtol"] = 1e-99
        newton.options["iprint"] = 2
        newton.options["maxiter"] = 50
        newton.options["solve_subsystems"] = True
        newton.options["max_sub_solves"] = 1000
        newton.options["reraise_child_analysiserror"] = False
        newton.options["err_on_non_converge"] = False

        ls = newton.linesearch = om.ArmijoGoldsteinLS()
        ls.options["maxiter"] = 3
        ls.options["rho"] = 0.75
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
        prob[pt + ".splitter.BPR"],
    )
    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC      BPR ", file=file, flush=True)
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f  %7.3f" % summary_data, file=file, flush=True)
    fs_names = [
        "fc.Fl_O",
        "inlet.Fl_O",
        "fan.Fl_O",
        "splitter.Fl_O1",
        "splitter.Fl_O2",
        "duct4.Fl_O",
        "lpc.Fl_O",
        "duct6.Fl_O",
        "hpc.Fl_O",
        "bld3.Fl_O",
        "burner.Fl_O",
        "hpt.Fl_O",
        "duct11.Fl_O",
        "lpt.Fl_O",
        "duct13.Fl_O",
        "core_nozz.Fl_O",
        "byp_bld.Fl_O",
        "duct15.Fl_O",
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
    shaft_names = ["hp_shaft", "lp_shaft"]
    shaft_full_names = [f"{pt}.{s}" for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)
    bleed_names = ["hpc", "bld3", "byp_bld"]
    bleed_full_names = [f"{pt}.{b}" for b in bleed_names]
    pyc.print_bleed(prob, bleed_full_names, file=file)


def map_plots(prob, pt):
    comp_names = ["fan", "lpc", "hpc"]
    comp_full_names = [f"{pt}.{c}" for c in comp_names]
    pyc.plot_compressor_maps(prob, comp_full_names)

    turb_names = ["lpt", "hpt"]
    turb_full_names = [f"{pt}.{c}" for c in turb_names]
    pyc.plot_turbine_maps(prob, turb_full_names)
