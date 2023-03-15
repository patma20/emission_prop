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
import numpy as np


# ==============================================================================
# Extension modules
# ==============================================================================


class TSEC(om.ExplicitComponent):
    def setup(self):
        self.add_input("TSFC", val=0.4, desc="Thrust-specific fuel consumption", units="lbm/(h*lbf)")
        self.add_input("LHV", val=18564, desc="Lower heating value of fuel", units="Btu/lbm")

        self.add_output("TSEC", val=800.0, desc="Thrust-specific energy consumption", units="Btu/(h*lbf)")

        self.declare_partials("TSEC", ["TSFC", "LHV"])

    def compute(self, inputs, outputs):
        outputs["TSEC"] = inputs["TSFC"] * inputs["LHV"]

    def compute_partials(self, inputs, J):
        J["TSEC", "TSFC"] = inputs["LHV"]
        J["TSEC", "LHV"] = inputs["TSFC"]


class NOxT4(om.ExplicitComponent):
    """
    Calculates NOx emissions of a combustor base on pre-combustor
    temperature and pressure and exhaust temperature.
    """

    def setup(self):
        self.add_input("P3", val=14.7, units="lbf/inch**2")
        self.add_input("T3", val=518.67, units="degR")
        self.add_input("T4", val=518.67, units="degR")

        self.add_output("EINOx", val=1.0, desc="NOx emissions index")
        self.declare_partials("EINOx", ["P3", "T3", "T4"])

    def compute(self, inputs, outputs):
        P3 = inputs["P3"]
        T3 = inputs["T3"]
        T4 = inputs["T4"]

        outputs["EINOx"] = 0.0042 * (P3 / 439.0) ** 0.37 * np.exp((T3 - 1471.0) / 345.0) * T4

    def compute_partials(self, inputs, J):
        P3 = inputs["P3"]
        T3 = inputs["T3"]
        T4 = inputs["T4"]

        J["EINOx", "P3"] = 0.0042 * 0.37 * (P3 / 439.0) ** (-0.63) * (1 / 439.0) * np.exp((T3 - 1471.0) / 345.0) * T4
        J["EINOx", "T3"] = 0.0042 * (P3 / 439.0) ** 0.37 * np.exp((T3 - 1471.0) / 345.0) * T4 / 345.0
        J["EINOx", "T4"] = 0.0042 * (P3 / 439.0) ** 0.37 * np.exp((T3 - 1471.0) / 345.0)


class NOxHum(om.ExplicitComponent):
    """
    Calculates NOx emissions of a combustor base on pre-combustor
    temperature and pressure and exhaust temperature.
    """

    def setup(self):
        self.add_input("P3", val=14.7, units="lbf/inch**2")
        self.add_input("T3", val=518.67, units="degR")
        # self.add_input("omega", val=0.0063, units="degR")

        self.add_output("EINOx", val=1.0, desc="NOx emissions index")
        self.declare_partials("EINOx", ["P3", "T3"], method="cs")

    def compute(self, inputs, outputs):
        P3 = inputs["P3"]
        T3 = inputs["T3"]
        H = 0.0063

        outputs["EINOx"] = 0.068 * P3 ** 0.5 * np.exp((T3 - 459.67) / 345.0) * np.exp(H * 0.0027114)

    # def compute_partials(self, inputs, J):
    #     P3 = inputs["P3"]
    #     T3 = inputs["T3"]

    #     J["EINOx", "P3"] = 0.0042 * 0.37 * (P3 / 439.0) ** (-0.63) * (1 / 439.0) * np.exp((T3 - 1471.0) / 345.0) * T4
    #     J["EINOx", "T3"] = 0.0042 * (P3 / 439.0) ** 0.37 * np.exp((T3 - 1471.0) / 345.0) * T4 / 345.0
    #     J["EINOx", "T4"] = 0.0042 * (P3 / 439.0) ** 0.37 * np.exp((T3 - 1471.0) / 345.0)


class SLSCorrelation(om.ExplicitComponent):
    """
    Calculate the NOx emissions index of a specific combustor given a
    correlation from the ICAO EDB emissions testing.
    """

    def setup(self):
        self.add_input("T3_SLS", val=1600.0, desc="Burner inlet temperature at SLS", units="degK")
        self.add_input("P3_SLS", val=100.0, desc="Burner inlet pressure at SLS", units="lbf/inch**2")
        self.add_output("EINOx_SLS", val=10.0, desc="NOx emissions index at SLS")

        self.declare_partials("EINOx_SLS", ["T3_SLS", "P3_SLS"], method="cs")

    def compute(self, inputs, outputs):
        T3 = inputs["T3_SLS"]
        P3 = inputs["P3_SLS"]

        outputs["EINOx_SLS"] = P3 ** 0.4 * (
            (6.26e-8 * T3 ** 3) - (0.000117 * T3 ** 2) + (0.074 * T3) - 15.04
        )  # MIT NOx correlation for CFM56-5B3 engine


class P3T3(om.ExplicitComponent):
    """
    Calculate the NOx emissions index using the P3T3 correlation method.
    """

    def setup(self):
        self.add_input("P3_OD", val=1.0, desc="Burner inlet pressure off-design", units="lbf/inch**2")
        self.add_input("P3_SLS", val=1.0, desc="Burner inlet pressure SLS", units="lbf/inch**2")
        self.add_input("FAR_OD", val=0.3, desc="Fuel-to-air-ratio at off-design")
        self.add_input("FAR_SLS", val=0.3, desc="Fuel-to-air-ratio at SLS")
        self.add_input("H", val=7.0, desc="Humidity factor exponent")
        self.add_input("EINOx_SLS", val=10.0, desc="NOx emissions index at SLS")

        self.add_output("EINOx_OD", val=15.0, desc="NOx emissions index at off-design")

        self.declare_partials("EINOx_OD", ["P3_OD", "P3_SLS", "FAR_OD", "FAR_SLS", "H", "EINOx_SLS"])

        self.m = 0.0
        self.n = 0.4

    def compute(self, inputs, outputs):
        # --- Define exponents ---
        n = self.n
        m = self.m

        # --- Get inputs ---
        P3_OD = inputs["P3_OD"]
        P3_SLS = inputs["P3_SLS"]
        FAR_OD = inputs["FAR_OD"]
        FAR_SLS = inputs["FAR_SLS"]
        H = inputs["H"]
        EINOx_SLS = inputs["EINOx_SLS"]

        # --- Solve for EINOx at off-design ---
        outputs["EINOx_OD"] = EINOx_SLS * ((P3_OD / P3_SLS) ** n) * ((FAR_OD / FAR_SLS) ** m) * np.exp(H)

    def compute_partials(self, inputs, partials):
        # --- Set exponents ---
        n = self.n
        m = self.m

        # --- Get inputs ---
        P3_OD = inputs["P3_OD"]
        P3_SLS = inputs["P3_SLS"]
        FAR_OD = inputs["FAR_OD"]
        FAR_SLS = inputs["FAR_SLS"]
        H = inputs["H"]
        EINOx_SLS = inputs["EINOx_SLS"]

        # --- Partials of EINOx_OD w.r.t all inputs ---
        partials["EINOx_OD", "P3_OD"] = (
            EINOx_SLS * n * ((P3_OD / P3_SLS) ** n) * ((FAR_OD / FAR_SLS) ** m) * np.exp(H)
        ) / P3_OD

        partials["EINOx_OD", "P3_SLS"] = (
            -(EINOx_SLS * n * ((P3_OD / P3_SLS) ** n) * ((FAR_OD / FAR_SLS) ** m) * np.exp(H)) / P3_SLS
        )

        partials["EINOx_OD", "EINOx_SLS"] = ((P3_OD / P3_SLS) ** n) * ((FAR_OD / FAR_SLS) ** m) * np.exp(H)

        partials["EINOx_OD", "FAR_OD"] = (
            EINOx_SLS * m * ((P3_OD / P3_SLS) ** n) * ((FAR_OD / FAR_SLS) ** m) * np.exp(H)
        ) / FAR_OD

        partials["EINOx_OD", "FAR_SLS"] = (
            -(EINOx_SLS * m * ((P3_OD / P3_SLS) ** n) * ((FAR_OD / FAR_SLS) ** m) * np.exp(H)) / FAR_SLS
        )

        partials["EINOx_OD", "H"] = EINOx_SLS * ((P3_OD / P3_SLS) ** n) * ((FAR_OD / FAR_SLS) ** m) * np.exp(H)


class EINOx(om.Group):
    """
    Calculates the NOx emissions index of an off-design point given engine
    states and humidity conditions (g water/g dry air) at SLS and specified off-design point.
    """

    def setup(self):
        self.add_subsystem("SLS_NOx_calc", SLSCorrelation(), promotes_inputs=["T3_SLS"], promotes_outputs=["EINOx_SLS"])
        self.add_subsystem(
            "humidity_calc",
            om.ExecComp("H=19.0 * (h_SLS - h_OD)", H={"val": 7.0}, h_SLS={"val": 0.6}, h_OD={"val": 0.4}),
            promotes_inputs=["h_SLS", "h_OD"],
            promotes_outputs=["H"],
        )
        self.add_subsystem(
            "P3T3_calc",
            P3T3(),
            promotes_inputs=["P3_OD", "P3_SLS", "FAR_OD", "FAR_SLS", "H", "EINOx_SLS"],
            promotes_outputs=["EINOx_OD"],
        )
