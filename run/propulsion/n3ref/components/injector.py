""" Class definition for Combustor."""

# import numpy as np

import openmdao.api as om

from pycycle.thermo.thermo import Thermo, ThermoAdd

from pycycle.thermo.cea.species_data import janaf

from pycycle.elements.duct import PressureLoss

from pycycle.flow_in import FlowIn
from pycycle.passthrough import PassThrough
from pycycle.element_base import Element


class Injector(Element):
    """
    A injector that adds a reactant to an incoming flow mixture

    --------------
    Flow Stations
    --------------
    Fl_I
    Fl_O

    -------------
    Design
    -------------
        inputs
        --------
        WAR
        dPqP
        MN

        outputs
        --------
        Wreact


    -------------
    Off-Design
    -------------
        inputs
        --------
        WAR
        dPqP
        area

        outputs
        --------
        Wreact

    """

    def initialize(self):

        self.options.declare("statics", default=True, desc="If True, calculate static properties.")
        self.options.declare(
            "reactant",
            default=False,
            types=(bool, str),
            desc="If False, flow matches base composition. If a string, then that reactant "
            "is mixed into the flow at at the ratio set by the `mix_ratio` input",
        )
        self.options.declare(
            "mix_name",
            default="mix",
            desc="The name of the input that governs the mix of the reactant to the primary flow",
        )

        self.options.declare("spec", default=janaf, desc="Thermodynamic data set for flow.", recordable=False)

        super().initialize()

    def pyc_setup_output_ports(self):

        thermo_method = self.options["thermo_method"]
        thermo_data = self.options["thermo_data"]
        reactant = self.options["reactant"]
        # spec = self.options["spec"]

        self.thermo_add_comp = ThermoAdd(
            method=thermo_method,
            mix_mode="flow",
            thermo_kwargs={
                "spec": thermo_data,
                "inflow_composition": self.Fl_I_data["Fl_I"],
                "mix_composition": thermo_data.reactants[reactant],
            },
        )

        self.copy_flow(self.thermo_add_comp, "Fl_O")

    def setup(self):
        thermo_method = self.options["thermo_method"]
        thermo_data = self.options["thermo_data"]

        # inflow_composition = self.Fl_I_data["Fl_I"]
        air_react_composition = self.Fl_O_data["Fl_O"]
        design = self.options["design"]
        statics = self.options["statics"]

        mix_name = self.options["mix_name"]

        # Compute reactant-air mixture ratio
        # self.add_subsystem(
        #     "MR", MixtureRatio(), promotes=["mdot_r", ("mdot_a", "Fl_I:stat:W"), ("mix_ratio", "mix:ratio")]
        # )

        # Create injector flow station
        in_flow = FlowIn(fl_name="Fl_I")
        self.add_subsystem("in_flow", in_flow, promotes=["Fl_I:tot:*", "Fl_I:stat:*"])

        # Create output_ports instance
        self.add_subsystem(
            "mix_react",
            self.thermo_add_comp,
            promotes=[
                "Fl_I:stat:W",
                (f"{mix_name}:composition", "mix_composition"),
                "Fl_I:tot:composition",
                "Fl_I:tot:h",
                f"{mix_name}:h",
                f"{mix_name}:W",
                "Wout",
            ],
        )

        # Pressure loss
        prom_in = [("Pt_in", "Fl_I:tot:P"), "dPqP"]
        self.add_subsystem("p_loss", PressureLoss(), promotes_inputs=prom_in)

        # Calculate new composition flow station properties
        real_flow = Thermo(
            mode="total_hP",
            fl_name="Fl_O:tot",
            method=thermo_method,
            thermo_kwargs={"composition": air_react_composition, "spec": thermo_data},
        )
        # prom_in = [("composition", "Fl_I:tot:composition"), ("T", "Fl_I:tot:T"), ("P", "Fl_I:tot:P")]
        self.add_subsystem("real_flow", real_flow, promotes_outputs=["Fl_O:*"])  # promotes_inputs=prom_in
        self.connect("mix_react.mass_avg_h", "real_flow.h")
        self.connect("mix_react.composition_out", "real_flow.composition")
        self.connect("p_loss.Pt_out", "real_flow.P")

        # Calculate vitiated flow station properties
        # vit_flow = Thermo(
        #     mode="total_hP",
        #     fl_name="Fl_O:tot",
        #     method=thermo_method,
        #     thermo_kwargs={"composition": air_react_composition, "spec": thermo_data},
        # )
        # self.add_subsystem("vitiated_flow", vit_flow, promotes_outputs=["Fl_O:*"])
        # self.connect("mix_react.mass_avg_h", "vitiated_flow.h")
        # self.connect("mix_react.composition_out", "vitiated_flow.composition")
        # self.connect("p_loss.Pt_out", "vitiated_flow.P")

        if statics:
            if design:
                # Calculate static properties.

                out_stat = Thermo(
                    mode="static_MN",
                    fl_name="Fl_O:stat",
                    method=thermo_method,
                    thermo_kwargs={"composition": air_react_composition, "spec": thermo_data},
                )
                prom_in = ["MN"]
                prom_out = ["Fl_O:stat:*"]
                self.add_subsystem("out_stat", out_stat, promotes_inputs=prom_in, promotes_outputs=prom_out)

                self.connect("mix_react.composition_out", "out_stat.composition")
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
                    thermo_kwargs={"composition": air_react_composition, "spec": thermo_data},
                )
                prom_in = ["area"]
                prom_out = ["Fl_O:stat:*"]
                self.add_subsystem("out_stat", out_stat, promotes_inputs=prom_in, promotes_outputs=prom_out)
                self.connect("mix_react.composition_out", "out_stat.composition")

                self.connect("Fl_O:tot:S", "out_stat.S")
                self.connect("Fl_O:tot:h", "out_stat.ht")
                self.connect("Fl_O:tot:P", "out_stat.guess:Pt")
                self.connect("Fl_O:tot:gamma", "out_stat.guess:gamt")
                self.connect("Wout", "out_stat.W")

        else:
            self.add_subsystem("W_passthru", PassThrough("Wout", "Fl_O:stat:W", 1.0, units="lbm/s"), promotes=["*"])

        super().setup()


if __name__ == "__main__":

    p = om.Problem()
    p.model = om.Group()
    p.model.add_subsystem("comp", Injector(), promotes=["*"])

    p.model.add_subsystem(
        "d1", om.IndepVarComp("Fl_I:stat:W", val=1.0, units="lbm/s", desc="weight flow"), promotes=["*"]
    )
    p.model.add_subsystem("d2", om.IndepVarComp("mix:ratio", val=0.2, desc="Reactant to air ratio"), promotes=["*"])
    p.model.add_subsystem(
        "d3", om.IndepVarComp("Fl_I:tot:h", val=1.0, units="Btu/lbm", desc="total enthalpy"), promotes=["*"]
    )
    p.model.add_subsystem(
        "d4", om.IndepVarComp("fuel_Tt", val=518.0, units="degR", desc="fuel temperature"), promotes=["*"]
    )

    p.setup(check=False, force_alloc_complex=True)
    p.run_model()

    p.check_partials(compact_print=True, method="cs")
