""" Class definition for a Extractor."""

import numpy as np

import openmdao.api as om

# from pycycle.thermo.cea import species_data
from pycycle.thermo.thermo import Thermo
from pycycle.flow_in import FlowIn
from pycycle.passthrough import PassThrough
from pycycle.element_base import Element
from pycycle.constants import CEA_AIR_COMPOSITION
from pycycle.thermo.cea.species_data import Properties, janaf, wet_air
from pycycle.elements.duct import PressureLoss


class ThermoSub(om.ExplicitComponent):
    """
    ThermoSub calculates a new composition given inflow, a species to subtract, and a extraction ratio.
    """

    def initialize(self):
        self.options.declare("spec", default=janaf, desc=("Thermodynamic data set for flow."), recordable=False)
        self.options.declare("inflow_composition", default=None, desc="composition present in the flow")

    def setup(self):
        spec = self.options["spec"]
        inflow_composition = self.options["inflow_composition"]

        if inflow_composition is None:
            inflow_composition = CEA_AIR_COMPOSITION

        inflow_thermo = Properties(spec, init_elements=inflow_composition)
        self.products = inflow_thermo.products
        self.inflow_composition = inflow_thermo.elements
        self.inflow_wt_mole = inflow_thermo.element_wt
        self.num_inflow_composition = len(self.inflow_composition)

        # inputs
        self.add_input("Fl_I:stat:W", val=0.0, desc="weight flow", units="lbm/s")
        self.add_input("Fl_I:tot:composition", val=inflow_thermo.b0, desc="incoming flow composition")
        self.add_input("n", val=np.ones(17), shape=17, desc="molar concetration of incoming flow")
        self.add_input("w_frac", val=0.01, desc="fraction of water from incoming flow to extract")

        # outputs
        self.add_output("Wout", shape=1, units="lbm/s", desc="main massflow out")  # add initial vals
        self.add_output("W_water", shape=1, units="lbm/s", desc="water massflow out")
        self.add_output("composition_water", val=inflow_thermo.b0)
        self.add_output("composition_out", val=inflow_thermo.b0)

        self.declare_partials("*", "*", method="cs")
        # NOTE: Due to some complexity from python vectorization,
        # computing partials manually is tricky and will not be much faster than pure CS

    def compute(self, inputs, outputs):

        W = inputs["Fl_I:stat:W"]  # incoming flow rate
        n = inputs["n"]  # moles of products of imcoming flow
        idx = self.products.index("H2O")
        n_h2o = n[idx]
        w_frac = inputs["w_frac"]
        Fl_I_tot_b0 = inputs["Fl_I:tot:composition"]  # elemental
        Fl_I_names = self.inflow_composition

        # copy the incoming flow into a correctly sized array for the outflow composition
        b0_out = np.array(Fl_I_tot_b0)

        sub_comp = 0 * b0_out
        sub_comp[Fl_I_names.index("H")] = 2 * n_h2o
        sub_comp[Fl_I_names.index("O")] = n_h2o

        sub_comp *= self.inflow_wt_mole
        b0_out *= self.inflow_wt_mole  # convert to mass units
        sub_comp /= np.sum(b0_out)
        b0_out /= np.sum(b0_out)  # scale to 1 kg
        sub_comp *= W  # scale to total mass flow
        sub_comp *= w_frac  # only subtract a fraction of water elements
        b0_out *= W  # scale to full mass flow

        b0_out -= sub_comp  # subtract water element concentration
        b0_out /= np.sum(b0_out)  # scale back to 1 kg
        w_water = np.sum(sub_comp)
        outputs["composition_out"] = b0_out / self.inflow_wt_mole
        outputs["Wout"] = W - w_water

        outputs["composition_water"] = sub_comp
        outputs["W_water"] = w_water


class WaterBleed(Element):
    """
    extract water from the incoming flow
    --------------
    Flow Stations
    --------------
    Fl_I -> primary input flow
    Fl_O -> primary output flow

    -------------
    Design
    -------------
        inputs
        --------
        w_frac
        MN
        dPqP

        outputs
        --------
        W_water
        Wout

    -------------
    Off-Design
    -------------
        inputs
        --------
        w_frac
        dPqP
        area

        outputs
        --------
        W_water
        Wout
    """

    def initialize(self):
        self.options.declare("statics", default=True, desc="If True, calculate static properties.")
        self.options.declare("design_water", default=False, types=bool, desc="If True, set DP as water injection.")

        self.default_des_od_conns = [
            # (design src, off-design target)
            ("Fl_O:stat:area", "area")
        ]

        super().initialize()

    def pyc_setup_output_ports(self):
        self.copy_flow(
            "Fl_I", "Fl_O"
        )  # since we are extracting only a fraction of water, copy flow element composition dictionary

    def setup(self):
        thermo_method = self.options["thermo_method"]  # will always be CEA for this component
        thermo_data = self.options["thermo_data"]  #
        statics = self.options["statics"]
        # design = self.options["design"]
        design_water = self.options["design_water"]
        composition = self.Fl_O_data["Fl_O"]  # dictionary of elements strings and associated elemental ratio

        # Compute equilibrium composition before extraction to get moles of H2O in flow
        init_flow = Thermo(
            mode="total_TP",
            fl_name="Fl_I:tot",
            method=thermo_method,
            thermo_kwargs={"composition": composition, "spec": thermo_data},
        )
        prom_in = [("composition", "Fl_I:tot:composition"), ("T", "Fl_I:tot:T"), ("P", "Fl_I:tot:P")]
        self.init_flow_data = self.add_subsystem("init_flow", init_flow, promotes_inputs=prom_in, promotes_outputs=[])

        # Create inlet flowstation
        flow_in = FlowIn(fl_name="Fl_I")
        self.add_subsystem("flow_in", flow_in, promotes=["Fl_I:tot:*", "Fl_I:stat:*"])

        # Create object to subtract water from flow
        thermo_sub_comp = ThermoSub(spec=thermo_data, inflow_composition=self.Fl_I_data["Fl_I"])

        # Create output_ports instance
        self.ext_sys = self.add_subsystem(
            "sub_flow",
            thermo_sub_comp,
            promotes=["Fl_I:stat:W", "Fl_I:tot:composition", "Wout", "W_water", "composition_out"],
        )

        # Pressure loss
        prom_in = [("Pt_in", "Fl_I:tot:P"), "dPqP"]
        self.add_subsystem("p_loss", PressureLoss(), promotes_inputs=prom_in, promotes_outputs=["Pt_out"])

        # Calculate total properties of equilibrium flow with updated composition
        updated_flow = Thermo(
            mode="total_TP",
            fl_name="Fl_O:tot",
            method=thermo_method,
            thermo_kwargs={"composition": composition, "spec": thermo_data},
        )
        prom_in = [("composition", "composition_out"), ("T", "Fl_I:tot:T"), ("P", "Pt_out")]
        self.add_subsystem("updated_flow", updated_flow, promotes_inputs=prom_in, promotes_outputs=["Fl_O:*"])

        if statics:
            if design_water:
                # Calculate static properties.
                out_stat = Thermo(
                    mode="static_MN",
                    fl_name="Fl_O:stat",
                    method=thermo_method,
                    thermo_kwargs={"composition": composition, "spec": thermo_data},
                )
                prom_in = ["MN"]
                prom_out = ["Fl_O:stat:*"]
                self.add_subsystem("out_stat", out_stat, promotes_inputs=prom_in, promotes_outputs=prom_out)

                self.connect("composition_out", "out_stat.composition")
                self.connect("Fl_O:tot:S", "out_stat.S")
                self.connect("Fl_O:tot:h", "out_stat.ht")
                self.connect("Fl_O:tot:P", "out_stat.guess:Pt")
                self.connect("Fl_O:tot:gamma", "out_stat.guess:gamt")
                self.connect("Wout", "out_stat.W")

            else:
                # Calculate static properties.
                out_stat = Thermo(
                    mode="static_A",
                    fl_name="Fl_O:stat",
                    method=thermo_method,
                    thermo_kwargs={"composition": composition, "spec": thermo_data},
                )
                prom_in = ["area"]
                prom_out = ["Fl_O:stat:*"]
                self.add_subsystem("out_stat", out_stat, promotes_inputs=prom_in, promotes_outputs=prom_out)

                self.connect("composition_out", "out_stat.composition")
                self.connect("Fl_O:tot:S", "out_stat.S")
                self.connect("Fl_O:tot:h", "out_stat.ht")
                self.connect("Fl_O:tot:P", "out_stat.guess:Pt")
                self.connect("Fl_O:tot:gamma", "out_stat.guess:gamt")
                self.connect("Wout", "out_stat.W")

        else:
            self.add_subsystem("W_passthru", PassThrough("Wout", "Fl_O:stat:W", 1.0, units="lbm/s"), promotes=["*"])

        super().setup()

    def configure(self):
        # Connect molar fraction, n, array to subsystem to determine amount of H2O in stream
        self.connect("init_flow.base_thermo.n", "sub_flow.n")
        return super().configure()


if __name__ == "__main__":

    p = om.Problem()
    p.model = om.Group()

    n_values = np.array(
        [
            3.15643098e-04,
            1.00000000e-10,
            1.53953831e-03,
            1.00000000e-10,
            1.00000000e-10,
            1.00000000e-10,
            1.59923925e-03,
            1.00000000e-10,
            1.00000000e-10,
            1.00000000e-10,
            9.18856891e-09,
            4.30532475e-09,
            1.00000000e-10,
            2.63178529e-02,
            1.00000000e-10,
            1.00000000e-10,
            4.79895474e-03,
        ]
    )
    des_vars = p.model.add_subsystem("des_vars", om.IndepVarComp(), promotes=["*"])
    des_vars.add_output("Fl_I:stat:W", 33.65599979, units="lbm/s")
    des_vars.add_output("Fl_I:tot:T", 518.67, units="degR")
    des_vars.add_output("Fl_I:tot:P", 14.696, units="psi")
    des_vars.add_output("Fl_I:tot:composition", [0.00031564, 0.00153954, 0.00319848, 0.05263572, 0.01427624])
    des_vars.add_output("w_frac", 0.99, units=None)
    des_vars.add_output("n_water", 0.0012402370022408862, units=None)
    des_vars.add_output("n", n_values, units=None)

    inflow_comp = {"N": 0.0539157698, "O": 1.0, "Ar": 0.000323319235, "C": 2.0, "H": 4.0044}
    spec = wet_air

    thermo_kwargs = {"spec": spec, "inflow_composition": inflow_comp}

    p.model.add_subsystem("therm_sub", ThermoSub(**thermo_kwargs), promotes=["*"])

    # init_flow = Thermo(
    #     mode="total_TP", fl_name="Fl_I:tot", method="CEA", thermo_kwargs={"composition": inflow_comp, "spec": spec}
    # )
    # prom_in = [("composition", "Fl_I:tot:composition"), ("T", "Fl_I:tot:T"), ("P", "Fl_I:tot:P")]
    # init_flow_data = p.model.add_subsystem("init_flow", init_flow, promotes_inputs=prom_in, promotes_outputs=[])

    p.setup()
    # p.model.options["thermo_method"] = "CEA"
    # p.model.options["thermo_data"] = pyc.species_data.wet_air  # use this spec data set due to numerical issues
    # p.model.therm_sub.options["spec"] = wet_air
    # p.model.therm_sub.options["inflow_composition"] = {
    #     "N": 0.0539157698,
    #     "O": 1.0,
    #     "Ar": 0.000323319235,
    #     "C": 2.0,
    #     "H": 4.0044,
    # }

    # p.setup()
    p.run_model()
    # p.check_partials(compact_print=True, show_only_incorrect=False, method="fd")

    # names = init_flow_data.base_thermo.thermo.products
    # compounds = init_flow_data.base_thermo.chem_eq._outputs["n"]

    # n_h2o = compounds[names.index("H2O")]

    # print(n_h2o)
    # print(compounds)
    # print(names)

    print("Flow In:", p["Fl_I:stat:W"])
    print("Flow Out:", p["Wout"])
    print("Water out:", p["W_water"])
    print("In Comp:", p["Fl_I:tot:composition"])
    print("Out Comp:", p["composition_out"])

    # print("W", p["Fl_I:stat:W"], p["Fl_O:stat:W"], p["test1:stat:W"], p["test2:stat:W"])
    # print("T", p["Fl_I:tot:T"], p["Fl_O:tot:T"], p["test1:tot:T"], p["test2:tot:T"])
    # print("P", p["Fl_I:tot:P"], p["Fl_O:tot:P"], p["test1:tot:P"], p["test2:tot:P"])
    # p.check_partials()
