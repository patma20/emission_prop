# --- Python 3.10 ---
"""
A holding place for OpenMDAO components that don't belong in any particular script.
"""
# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import openmdao.api as om
import warnings
from thermo import Mixture
from openmdao.api import Group, IndepVarComp
from openconcept.utilities.dvlabel import DVLabel
from openconcept.thermal.heat_exchanger import (
    OffsetStripFinGeometry,
    HydraulicDiameterReynoldsNumber,
    OffsetStripFinData,
    NusseltFromColburnJ,
    ConvectiveCoefficient,
    FinEfficiency,
    UAOverall,
    NTUMethod,
    CrossFlowNTUEffectiveness,
    NTUEffectivenessActualHeatTransfer,
    OutletTemperatures,
    PressureDrop,
)


# ==============================================================================
# Extension modules
# ==============================================================================


class dPqP_comp(om.ExplicitComponent):
    """
    Connects pressure drop to change in pressure
    """

    def setup(self):
        self.add_input("delta_p", val=0.001, units="lbf/inch**2")
        self.add_input("Pt_in", val=1.0, units="lbf/inch**2")

        self.add_output("dPqP", val=0.01)

    def setup_partials(self):
        # self.declare_partials("*", "*", method="fd")
        self.declare_partials(["dPqP"], ["Pt_in", "delta_p"])

    def compute(self, inputs, outputs):
        outputs["dPqP"] = -inputs["delta_p"] / inputs["Pt_in"]

    def compute_partials(self, inputs, J):
        J["dPqP", "delta_p"] = -1 / inputs["Pt_in"]
        J["dPqP", "Pt_in"] = inputs["delta_p"] / inputs["Pt_in"] ** 2


class area_con(om.ExplicitComponent):
    """
    Area constraint for duct and HX areas. Must be equal to 1.
    """

    def setup(self):
        self.add_input("HX_fa", val=1.0, units="inch**2")
        self.add_input("HX_duct_area", val=1.0, units="inch**2")

        self.add_output("area_con", val=1.0)

    def setup_partials(self):
        self.declare_partials(["area_con"], ["HX_fa", "HX_duct_area"])

    def compute(self, inputs, outputs):
        outputs["area_con"] = inputs["HX_fa"] / inputs["HX_duct_area"]

    def compute_partials(self, inputs, J):
        J["area_con", "HX_fa"] = 1.0 / inputs["HX_duct_area"]
        J["area_con", "HX_duct_area"] = -inputs["HX_fa"] / inputs["HX_duct_area"] ** 2


class heat_comp(om.ExplicitComponent):
    """
    Ideal heat transfer based on mdot, T_c, T_h, and specific heat

    q_in = Cp * mdot * (T_h - T_c)
    """

    def setup(self):
        self.add_input("q_in", val=1.0, units="W")
        self.add_input("T_c", val=1.0, units="degK")
        self.add_input("mdot", val=1.0, units="kg/s")
        self.add_input("Cp", val=1.0, units="J/kg/degK")

        self.add_output("T_h", val=1.0, units="degK")

    def setup_partials(self):
        self.declare_partials(["T_h"], ["q_in", "T_c", "mdot", "Cp"])

    def compute(self, inputs, outputs):
        outputs["T_h"] = inputs["q_in"] / (inputs["Cp"] * inputs["mdot"]) + inputs("T_c")

    def compute_partials(self, inputs, J):
        J["T_h", "q_in"] = 1.0 / (inputs["Cp"] * inputs["mdot"]) + inputs("T_c")
        J["T_h", "Cp"] = -inputs["q_in"] / (inputs["Cp"] ** 2 * inputs["mdot"]) + inputs("T_c")
        J["T_h", "mdot"] = -inputs["q_in"] / (inputs["Cp"] * inputs["mdot"] ** 2) + inputs("T_c")
        J["T_h", "T_c"] = inputs["q_in"] / (inputs["Cp"] * inputs["mdot"]) + 1


class GetFluidProps(om.ExplicitComponent):
    """
    Retrieves fluid thermodynamic and transport properties based on temperature and pressure
    Inputs
    ------
    T : float
        Temperature of fluid (scalar, K)
    P : float
        Pressure of fluid (scalar, Pa)
    """

    def initialize(self):
        self.options.declare("fluid_species", default="air", desc="Species of fluid of which to get properties")

    def setup(self):
        self.add_input("T", val=300, units="K")
        self.add_input("P", val=101325, units="Pa")

        self.add_output("cp", val=1005, units="J/kg/K")
        self.add_output("k", val=0.02596, units="W/m/K")
        self.add_output("mu", val=1.789e-5, units="kg/m/s")
        self.add_output("rho", val=1020, units="kg/m**3")

        self.declare_partials("*", "*", method="fd")
        # self.set_check_partial_options(wrt=["T", "P"], method="fd")

    def compute(self, inputs, outputs):
        fluid_species = self.options["fluid_species"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fluid = Mixture(fluid_species, T=inputs["T"], P=inputs["P"])

            outputs["k"] = fluid.k
            outputs["mu"] = fluid.mu
            # outputs["rho"] = fluid.rho
            # outputs["cp"] = fluid.Cp


class HeatExchanger(Group):
    """
    A heat exchanger model for use with the duct models
    Note that there are many design variables defined as dvs which could be varied
    in optimization.
    Inputs
    ------
    mdot_cold : float
        Mass flow rate of the cold side (air) (vector, kg/s)
    T_in_cold : float
        Inflow temperature of the cold side (air) (vector, K)
    rho_cold : float
        Inflow density of the cold side (air) (vector, kg/m**3)
    mdot_hot : float
        Mass flow rate of the hot side (liquid) (vector, kg/s)
    T_in_hot : float
        Inflow temperature of the hot side (liquid) (vector, kg/s)
    rho_hot : float
        Inflow density of the hot side (liquid) (vector, kg/m**3)
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of analysis points")

    def setup(self):
        nn = self.options["num_nodes"]

        iv = self.add_subsystem("dv", IndepVarComp(), promotes_outputs=["*"])

        # General HX properties
        iv.add_output("case_thickness", val=2.0, units="mm")
        iv.add_output("fin_thickness", val=0.102, units="mm")
        iv.add_output("plate_thickness", val=0.2, units="mm")
        iv.add_output("material_k", val=190, units="W/m/K")
        iv.add_output("material_rho", val=2700, units="kg/m**3")

        # Cold side of HX properties
        iv.add_output("channel_height_cold", val=14, units="mm")
        iv.add_output("channel_width_cold", val=1.35, units="mm")
        iv.add_output("fin_length_cold", val=6, units="mm")
        # iv.add_output("cp_cold", val=1005, units="J/kg/K")
        # iv.add_output("k_cold", val=0.02596, units="W/m/K")
        # iv.add_output("mu_cold", val=1.789e-5, units="kg/m/s")

        # Hot side of HX properties
        iv.add_output("channel_height_hot", val=1, units="mm")
        iv.add_output("channel_width_hot", val=1, units="mm")
        iv.add_output("fin_length_hot", val=6, units="mm")
        iv.add_output("cp_hot", val=3801, units="J/kg/K")
        iv.add_output("k_hot", val=0.405, units="W/m/K")
        iv.add_output("mu_hot", val=1.68e-3, units="kg/m/s")

        dvlist = [
            ["ac|propulsion|thermal|hx|n_wide_cold", "n_wide_cold", 200, None],
            ["ac|propulsion|thermal|hx|n_long_cold", "n_long_cold", 3, None],
            ["ac|propulsion|thermal|hx|n_tall", "n_tall", 15, None],
        ]

        self.add_subsystem("dvpassthru", DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])

        self.add_subsystem("osfgeometry", OffsetStripFinGeometry(), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "redh", HydraulicDiameterReynoldsNumber(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("osfdata", OffsetStripFinData(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("nusselt", NusseltFromColburnJ(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "convection", ConvectiveCoefficient(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("finefficiency", FinEfficiency(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("ua", UAOverall(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("ntu", NTUMethod(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem(
            "effectiveness", CrossFlowNTUEffectiveness(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem(
            "heat", NTUEffectivenessActualHeatTransfer(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"]
        )
        self.add_subsystem("t_out", OutletTemperatures(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
        self.add_subsystem("delta_p", PressureDrop(num_nodes=nn), promotes_inputs=["*"], promotes_outputs=["*"])
