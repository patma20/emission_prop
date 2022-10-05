# import sys
# import numpy as np

import openmdao.api as om

import pycycle.api as pyc

# import pycycle.constants as con

from injector import Injector


class Duct(pyc.Cycle):
    def initialize(self):
        super().initialize()

    def setup(self):

        self.options["thermo_method"] = "CEA"
        self.options["thermo_data"] = pyc.species_data.wet_air

        # self.add_subsystem("fc", pyc.FlightConditions())
        self.add_subsystem(
            "fc", pyc.FlightConditions(composition=pyc.CEA_AIR_COMPOSITION, reactant="Water", mix_ratio_name="WAR")
        )
        inlet = self.add_subsystem("inlet", pyc.Inlet())
        inject = self.add_subsystem("inject", Injector(reactant="Water", mix_ratio_name="mix:ratio"))
        duct = self.add_subsystem("duct", pyc.Duct())
        nozz = self.add_subsystem("nozz", pyc.Nozzle(nozzType="CV", lossCoef="Cv"))
        self.add_subsystem("perf", pyc.Performance(num_nozzles=1, num_burners=0))

        # indvars = self.add_subsystem("thermal_params", om.IndepVarComp(), promotes_outputs=["*"])
        # indvars.add_output("heat_load", 50.0, units="W")

        # Thermodynamic connections
        # self.connect("heat_load", "duct.Q_dot")

        # Connnect nozzle exhaust to freestream static conditions
        self.connect("fc.Fl_O:stat:P", "nozz.Ps_exhaust")

        # Connect outputs to pefromance element
        self.connect("inlet.Fl_O:tot:P", "perf.Pt2")
        self.connect("inlet.F_ram", "perf.ram_drag")
        self.connect("nozz.Fg", "perf.Fg_0")

        self.pyc_connect_flow("fc.Fl_O", "inlet.Fl_I", connect_w=False)
        # self.pyc_connect_flow("fc.Fl_O", "inlet.Fl_I")
        self.pyc_connect_flow("inlet.Fl_O", "inject.Fl_I")
        self.pyc_connect_flow("inject.Fl_O", "duct.Fl_I")
        # self.pyc_connect_flow("inlet.Fl_O", "duct.Fl_I")
        self.pyc_connect_flow("duct.Fl_O", "nozz.Fl_I")

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


class MPDuct(pyc.MPCycle):
    def setup(self):

        # Create design instance of model
        self.pyc_add_pnt("DESIGN", Duct(thermo_method="CEA"))

        self.set_input_defaults("DESIGN.fc.W", 400.0, units="lbm/s"),
        self.set_input_defaults("DESIGN.fc.alt", 37000.0, units="ft"),
        self.set_input_defaults("DESIGN.fc.MN", 0.4),
        self.set_input_defaults("DESIGN.inlet.MN", 0.350),
        self.set_input_defaults("DESIGN.inject.MN", 0.20),

        self.pyc_add_cycle_param("inject.dPqP", 0.03)
        self.pyc_add_cycle_param("nozz.Cv", 0.99)
        self.pyc_add_cycle_param("fc.WAR", 0.001)
        # self.pyc_add_cycle_param("inject.mix:ratio", 0.01)
        self.pyc_add_cycle_param("inject.mdot_r", 10.0, units="lbm/s")

        # self.od_pts = ["OD1"]
        # self.od_MNs = [0.000001]
        # self.od_alts = [0, 0]
        # self.od_pwrs = [11000.0]

        # for i, pt in enumerate(self.od_pts):
        #     self.pyc_add_pnt(pt, WetTurbojet(design=False, thermo_method="CEA"))

        #     self.set_input_defaults(pt + ".fc.MN", self.od_MNs[i]),
        #     self.set_input_defaults(pt + ".fc.alt", self.od_alts[i], units="ft"),
        #     self.set_input_defaults(pt + ".balance.rhs:FAR", self.od_pwrs[i], units="lbf")

        # self.pyc_use_default_des_od_conns()

        # self.pyc_connect_des_od("nozz.Throat:stat:area", "balance.rhs:W")

        super().setup()


if __name__ == "__main__":

    import time

    prob = om.Problem()
    prob.model = MPDuct()

    prob.setup()

    # Define the design point
    # prob.model.set_input_defaults("fc.W", 820.44097898, units="lbm/s")
    # prob.model.set_input_defaults("fc.alt", 35000.0, units="ft")
    # prob.model.set_input_defaults("fc.MN", 0.4)
    # prob.model.set_input_defaults("fc.mix:ratio", 0.001)
    # prob.model.set_input_defaults("inject.mix:ratio", 0.001)
    # prob.model.set_input_defaults("inlet.MN", 0.35),
    # prob.model.set_input_defaults("inlet.ram_recovery", 0.9980)
    # prob.model.set_input_defaults("duct.MN", 0.3)
    # prob.model.set_input_defaults("duct.dPqP", 0.0100)
    # prob.model.set_input_defaults("nozz.Cv", 0.9999),

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)
    prob.run_model()

    # for pt in ["TOC"] + prob.model.od_pts:
    #     viewer(prob, pt)

    # print("Diameter", prob["TOC.fan_dia.FanDia"][0])
    # print("ER", prob["CRZ.ext_ratio.ER"])
    # print("EINOx", prob["CRZ.NOx.EINOx"])
    # print("Composition", prob["inlet.Fl_I:tot:composition"])
    # print("Composition", prob["inlet.Fl_I:tot:T"])

    # print("prods", prob.model.duct.Fl_O_data["Fl_O"])  # Fl_I_data.inflow_composition
    # print("prods", prob.model.duct.Fl_O_data)  # Fl_I_data.inflow_composition
    # print("prods", vars(prob.model.duct.real_flow.base_thermo.thermo))  # Fl_I_data.inflow_composition

    inlet_prod_names = prob.model.DESIGN.inlet.real_flow.base_thermo.thermo.products
    inlet_prod_concs = prob.model.DESIGN.inlet.real_flow.base_thermo.chem_eq._outputs["n"]

    inject_prod_names = prob.model.DESIGN.inject.vitiated_flow.base_thermo.thermo.products
    inject_prod_concs = prob.model.DESIGN.inject.vitiated_flow.base_thermo.chem_eq._outputs["n"]

    for i in range(len(inlet_prod_names)):
        if inlet_prod_names[i] == "H2O":
            print(inlet_prod_names[i], inlet_prod_concs[i])
        if inject_prod_names[i] == "H2O":
            print(inject_prod_names[i], inject_prod_concs[i])
        # if prod_names[i] == "N2":
        #     n2 = prod_concs[i]
        # if prod_names[i] == "O2":
        #     o2 = prod_concs[i]

    # print(n2 / o2)
    # print("sum", np.sum(prod_concs))

    # print("prods", prod_names)  # Fl_I_data.inflow_composition
    # print("prod comps", prod_concs)  # Fl_I_data.inflow_composition
    # print("prods", prob.model.duct.real_flow.base_thermo.thermo.prod_data)  # Fl_I_data.inflow_composition
    # print("prods", prob.model.duct.products)  # Fl_I_data.inflow_composition
    # print("Composition", prob["duct.Fl_O:tot:composition"])
    # print("Composition", prob["duct.Fl_O:tot:T"])

    # vals = con.CEA_AIR_COMPOSITION.values()
    # sum = 0
    # for i in vals:
    #     sum += i
    # print(sum)

    # print("time", time.time() - st)
