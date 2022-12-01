import sys

import openmdao.api as om

import pycycle.api as pyc

import numpy as np

import pickle as pkl

from injector import Injector
from extractor import WaterBleed


class WetTurbojet(pyc.Cycle):
    def setup(self):

        design = self.options["design"]

        # NOTE: DEFAULT TABULAR thermo doesn't include WAR, so must use CEA here
        # (or build your own thermo tables)

        self.options["thermo_method"] = "CEA"
        self.options["thermo_data"] = pyc.species_data.wet_air  # use this spec data set due to numerical issues

        # Add engine elements
        self.add_subsystem(
            "fc", pyc.FlightConditions(composition=pyc.CEA_AIR_COMPOSITION, reactant="Water", mix_ratio_name="WAR")
        )  # composition is always the nominal dry air comp, reactant is to add the Water data to the reactants,
        # mix_ratio_name is just a descriptive variable name for the reactant-air ratio

        inlet = self.add_subsystem("inlet", pyc.Inlet())
        inject = self.add_subsystem("inject", Injector(reactant="Water", mix_name="mix"))
        self.add_subsystem("comp", pyc.Compressor(map_data=pyc.AXI5), promotes_inputs=["Nmech"])

        self.add_subsystem("burner", pyc.Combustor(fuel_type="JP-7"))
        turb = self.add_subsystem("turb", pyc.Turbine(map_data=pyc.LPT2269), promotes_inputs=["Nmech"])
        extract = self.add_subsystem("extract", WaterBleed())
        self.add_subsystem("nozz", pyc.Nozzle(nozzType="CD", lossCoef="Cv"))
        self.add_subsystem("shaft", pyc.Shaft(num_ports=2), promotes_inputs=["Nmech"])
        self.add_subsystem("perf", pyc.Performance(num_nozzles=1, num_burners=1))

        # Connect flow stations
        self.pyc_connect_flow("fc.Fl_O", "inlet.Fl_I", connect_w=False)
        self.pyc_connect_flow("inlet.Fl_O", "inject.Fl_I")
        self.pyc_connect_flow("inject.Fl_O", "comp.Fl_I")
        # self.pyc_connect_flow("inlet.Fl_O", "comp.Fl_I")
        self.pyc_connect_flow("comp.Fl_O", "burner.Fl_I")
        self.pyc_connect_flow("burner.Fl_O", "turb.Fl_I")
        self.pyc_connect_flow("turb.Fl_O", "extract.Fl_I")
        self.pyc_connect_flow("extract.Fl_O", "nozz.Fl_I")

        # Connect turbomachinery elements to shaft
        self.connect("comp.trq", "shaft.trq_0")
        self.connect("turb.trq", "shaft.trq_1")

        # Connnect nozzle exhaust to freestream static conditions
        self.connect("fc.Fl_O:stat:P", "nozz.Ps_exhaust")

        # Connect outputs to pefromance element
        self.connect("inlet.Fl_O:tot:P", "perf.Pt2")
        self.connect("comp.Fl_O:tot:P", "perf.Pt3")
        self.connect("burner.Wfuel", "perf.Wfuel_0")
        self.connect("inlet.F_ram", "perf.ram_drag")
        self.connect("nozz.Fg", "perf.Fg_0")

        # Add balances for design and off-design
        balance = self.add_subsystem("balance", om.BalanceComp())

        if design:

            balance.add_balance("W", units="lbm/s", eq_units="lbf")
            self.connect("balance.W", "inlet.Fl_I:stat:W")
            self.connect("perf.Fn", "balance.lhs:W")

            balance.add_balance("FAR", eq_units="degR", lower=1e-4, val=0.017)
            self.connect("balance.FAR", "burner.Fl_I:FAR")
            self.connect("burner.Fl_O:tot:T", "balance.lhs:FAR")

            balance.add_balance("turb_PR", val=1.5, lower=1.001, upper=8, eq_units="hp", rhs_val=0.0)
            self.connect("balance.turb_PR", "turb.PR")
            self.connect("shaft.pwr_net", "balance.lhs:turb_PR")

        else:

            balance.add_balance("FAR", eq_units="lbf", lower=1e-4, val=0.3)
            self.connect("balance.FAR", "burner.Fl_I:FAR")
            self.connect("perf.Fn", "balance.lhs:FAR")

            balance.add_balance("Nmech", val=1.5, units="rpm", lower=500.0, eq_units="hp", rhs_val=0.0)
            self.connect("balance.Nmech", "Nmech")
            self.connect("shaft.pwr_net", "balance.lhs:Nmech")

            balance.add_balance("W", val=168.0, units="lbm/s", eq_units="inch**2")
            self.connect("balance.W", "inlet.Fl_I:stat:W")
            self.connect("nozz.Throat:stat:area", "balance.lhs:W")

        # Setup solver to converge engine
        self.set_order(
            ["balance", "fc", "inlet", "inject", "comp", "burner", "turb", "extract", "nozz", "shaft", "perf"]
        )

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options["atol"] = 1e-6
        newton.options["rtol"] = 1e-6
        newton.options["iprint"] = 2
        newton.options["maxiter"] = 15
        newton.options["solve_subsystems"] = True
        newton.options["max_sub_solves"] = 100
        newton.options["reraise_child_analysiserror"] = False
        newton.linesearch = om.BoundsEnforceLS()
        # newton.linesearch = ArmijoGoldsteinLS()
        # newton.linesearch.options['c'] = .0001
        newton.linesearch.options["bound_enforcement"] = "scalar"
        newton.linesearch.options["iprint"] = -1

        self.linear_solver = om.DirectSolver(assemble_jac=True)

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
    )

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC  ", file=file, flush=True)
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f" % summary_data, file=file, flush=True)

    fs_names = [
        "fc.Fl_O",
        "inlet.Fl_O",
        "inject.Fl_O",
        "comp.Fl_O",
        "burner.Fl_O",
        "turb.Fl_O",
        "extract.Fl_O",
        "nozz.Fl_O",
    ]
    fs_full_names = [f"{pt}.{fs}" for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    comp_names = ["comp"]
    comp_full_names = [f"{pt}.{c}" for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    pyc.print_burner(prob, [f"{pt}.burner"])

    turb_names = ["turb"]
    turb_full_names = [f"{pt}.{t}" for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    noz_names = ["nozz"]
    noz_full_names = [f"{pt}.{n}" for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

    shaft_names = ["shaft"]
    shaft_full_names = [f"{pt}.{s}" for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)


class MPWetTurbojet(pyc.MPCycle):
    def setup(self):

        # Create design instance of model
        self.pyc_add_pnt("DESIGN", WetTurbojet(thermo_method="CEA"))

        # self.connect("DESIGN.extract.W_water", "DESIGN.inject.mix:W")

        nonlinear = self.nonlinear_solver = om.NonlinearBlockGS()
        nonlinear.options["atol"] = 1e-6
        nonlinear.options["rtol"] = 1e-99
        nonlinear.options["iprint"] = 2
        nonlinear.options["maxiter"] = 15

        self.set_input_defaults("DESIGN.fc.alt", 0.0, units="ft"),
        self.set_input_defaults("DESIGN.fc.MN", 0.000001),
        self.set_input_defaults("DESIGN.balance.rhs:FAR", 2370.0, units="degR"),
        self.set_input_defaults("DESIGN.balance.rhs:W", 11800.0, units="lbf"),
        self.set_input_defaults("DESIGN.Nmech", 8070.0, units="rpm"),
        self.set_input_defaults("DESIGN.inlet.MN", 0.60),
        self.set_input_defaults("DESIGN.comp.MN", 0.20),
        self.set_input_defaults("DESIGN.burner.MN", 0.20),
        self.set_input_defaults("DESIGN.turb.MN", 0.4),
        self.set_input_defaults("DESIGN.extract.MN", 0.4),
        self.set_input_defaults("DESIGN.inject.MN", 0.60),
        # self.set_input_defaults("DESIGN.inject.MN", 0.60),
        self.set_input_defaults("DESIGN.extract.sub_flow.w_frac", 0.1),
        self.set_input_defaults("OD1.extract.sub_flow.w_frac", 0.1),

        self.pyc_add_cycle_param("burner.dPqP", 0.03)
        self.pyc_add_cycle_param("inject.dPqP", 0.03)
        self.pyc_add_cycle_param("extract.dPqP", 0.0)
        self.pyc_add_cycle_param("nozz.Cv", 0.99)
        self.pyc_add_cycle_param("fc.WAR", 0.0001)
        # self.pyc_add_cycle_param("extract.sub_flow.w_frac", 0.1)
        # self.pyc_add_cycle_param("inject.mix:W", 0.1, units="lbm/s")
        # self.pyc_add_cycle_param("inject.mix:h", 10.0, units="Btu/lbm")

        self.od_pts = ["OD1"]
        self.od_MNs = [0.000001]
        self.od_alts = [0, 0]
        self.od_pwrs = [11000.0]

        for i, pt in enumerate(self.od_pts):
            self.pyc_add_pnt(pt, WetTurbojet(design=False, thermo_method="CEA"))

            self.set_input_defaults(pt + ".fc.MN", self.od_MNs[i]),
            self.set_input_defaults(pt + ".fc.alt", self.od_alts[i], units="ft"),
            self.set_input_defaults(pt + ".balance.rhs:FAR", self.od_pwrs[i], units="lbf")

        self.pyc_use_default_des_od_conns()

        self.connect("DESIGN.extract.W_water", "DESIGN.inject.mix:W")
        self.connect("OD1.extract.W_water", "OD1.inject.mix:W")

        self.pyc_connect_des_od("nozz.Throat:stat:area", "balance.rhs:W")

        super().setup()


if __name__ == "__main__":

    import time

    prob = om.Problem()

    prob.model = mp_wet_turbojet = MPWetTurbojet()

    prob.setup()

    # Define the design point
    prob.set_val("DESIGN.comp.PR", 13.5),
    prob.set_val("DESIGN.comp.eff", 0.83),
    prob.set_val("DESIGN.turb.eff", 0.86),

    # Set initial guesses for balances
    prob["DESIGN.balance.FAR"] = 0.0175506829934
    prob["DESIGN.balance.W"] = 168.453135137
    prob["DESIGN.balance.turb_PR"] = 4.46138725662
    prob["DESIGN.fc.balance.Pt"] = 14.6955113159
    prob["DESIGN.fc.balance.Tt"] = 518.665288153
    prob["DESIGN.inject.mix:W"] = 0.001

    for i, pt in enumerate(mp_wet_turbojet.od_pts):
        prob[pt + ".balance.W"] = 166.073
        prob[pt + ".balance.FAR"] = 0.01680
        prob[pt + ".balance.Nmech"] = 8197.38
        prob[pt + ".fc.balance.Pt"] = 15.703
        prob[pt + ".fc.balance.Tt"] = 558.31
        prob[pt + ".turb.PR"] = 4.6690

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    sweep = False
    check_cons_mass = True
    check_species = True
    check_elemental = False
    view = False

    if sweep is False:
        prob.run_model()

    if sweep is True:
        n = 20
        w_frac = np.linspace(0.000001, 0.2, n)
        TSFC = np.zeros(n)

        for i, w in enumerate(w_frac):
            print(f"### Running Wfrac={w} ###")
            prob["DESIGN.extract.sub_flow.w_frac"] = w
            prob.run_model()
            TSFC[i] = prob.get_val("DESIGN.perf.TSFC")

        with open("../OUTPUT/wet_turbo_trends/WetTurbo_wfrac.pkl", "wb") as f:
            pkl.dump(np.vstack((w_frac, TSFC)), f)

    if view:
        viewer(prob, "DESIGN")

    # print("Water Fraction", prob.get_val("extract.sub_flow.w_frac"))
    # print("Total Flow In:", prob["DESIGN.inlet.Fl_O:stat:W"] + prob["DESIGN.burner.Wfuel"])
    # print("Total Flow Out:", prob["DESIGN.nozz.Fl_O:stat:W"])
    # print("Main Flow In:", prob["DESIGN.extract.Fl_I:stat:W"])
    # print("Main Flow Out:", prob["DESIGN.extract.Wout"])
    # print("Inject Flow In:", prob["DESIGN.inject.mix:W"])
    # print("Extract Flow Out:", prob["DESIGN.extract.W_water"])
    # print("In Comp:", prob["DESIGN.extract.Fl_I:tot:composition"])
    # print("Out Comp:", prob["DESIGN.extract.composition_out"])

    if check_cons_mass:
        inlet_air = prob["DESIGN.inlet.Fl_O:stat:W"]
        inject_water = prob["DESIGN.inject.mix:W"]
        extract_water = prob["DESIGN.extract.W_water"]
        fuel_flow = prob["DESIGN.burner.Wfuel"]
        compress_flow = prob["DESIGN.comp.Fl_O:stat:W"]
        turb_flow = prob["DESIGN.turb.Fl_O:stat:W"]
        extract_out = prob["DESIGN.extract.Fl_O:stat:W"]
        nozz_flow = prob["DESIGN.nozz.Fl_O:stat:W"]

        print("Inlet Air Flow (lbm/s)", inlet_air)
        print("Inject Water Flow (lbm/s)", inject_water)
        print("Extract Water Flow (lbm/s)", extract_water)
        print("Combustor Fuel Flow", fuel_flow)
        print("Nozzle Out Flow", nozz_flow)

        print("Injector mass flow check (mdot_air + mdot_water - mdot_out): ", inlet_air + inject_water - compress_flow)
        print(
            "Combustor mass flow check (mdot_air-water + mdot_fuel - mdot_out): ", compress_flow + fuel_flow - turb_flow
        )
        print(
            "Extractor mass flow check (mdot_air-water-fuel - mdot_water - mdot_out): ",
            turb_flow - extract_water - extract_out,
        )
        print("Total mass flow check (mdot_in + mdot_fuel - mdot_out): ", inlet_air - nozz_flow + fuel_flow)
        print("Water loop mass flow check (mdot_in - mdot_out): ", extract_water - inject_water)

    if check_elemental:
        print(prob.model.DESIGN.inlet.Fl_O_data["Fl_O"])
        print(prob.model.DESIGN.inject.Fl_O_data["Fl_O"])
        print(prob.model.DESIGN.extract.Fl_O_data["Fl_O"])
        print(prob.model.DESIGN.nozz.Fl_O_data["Fl_O"])

    if check_species:
        nozz_prod_names = prob.model.DESIGN.nozz.ideal_flow.base_thermo.thermo.products
        nozz_prod_concs = prob.model.DESIGN.nozz.ideal_flow.base_thermo.chem_eq._outputs["n"]

        turb_prod_names = prob.model.DESIGN.turb.real_flow.base_thermo.thermo.products
        turb_prod_concs = prob.model.DESIGN.turb.real_flow.base_thermo.chem_eq._outputs["n"]

        print(100 * "#")
        print("Turbine molar concentrations (before water extraction)")
        nozz_prod_concs /= np.sum(nozz_prod_concs)
        turb_prod_concs /= np.sum(turb_prod_concs)
        for n, e in zip(turb_prod_names, turb_prod_concs):
            if e < 10e-8:
                e = 0.0
            else:
                print(n, e)

        print(100 * "#")
        print("Nozzle molar concentrations (after water extraction)")
        for n, e in zip(nozz_prod_names, nozz_prod_concs):
            if e < 10e-8:
                e = 0.0
            else:
                print(n, e)

    print()
    print("time", time.time() - st)
