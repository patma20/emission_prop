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


class NOxT4(om.ExplicitComponent):
    """
    Calculates NOx emissions of a combustor base on pre-combustor
    temperature and pressure and exhaust temperature.
    """

    def setup(self):
        self.add_input("P3", val=14.7, units="lbf/inch**2")
        self.add_input("T3", val=518.67, units="degR")
        self.add_input("T4", val=518.67, units="degR")

        self.add_output("EINOx", val=10.0, desc="NOx emissions index")
        self.declare_partials("EINOx", ["P3", "T3", "T4"])

    def compute(self, inputs, outputs):
        P3 = inputs["P3"]
        T3 = inputs["T3"]
        T4 = inputs["T4"]

        outputs["EINOx"] = 0.0043 * (P3 / 439.0) ** 0.37 * np.exp((T3 - 1471.0) / 345.0) * T4

    def compute_partials(self, inputs, J):
        P3 = inputs["P3"]
        T3 = inputs["T3"]
        T4 = inputs["T4"]

        J["EINOx", "P3"] = 0.0043 * 0.37 * (P3 / 439.0) ** (-0.63) * (1 / 439.0) * np.exp((T3 - 1471.0) / 345.0) * T4
        J["EINOx", "T3"] = 0.0043 * (P3 / 439.0) ** 0.37 * np.exp((T3 - 1471.0) / 345.0) * T4 / 345.0
        J["EINOx", "T4"] = 0.0043 * (P3 / 439.0) ** 0.37 * np.exp((T3 - 1471.0) / 345.0)


class NOxFit(om.ExplicitComponent):
    """
    Calculate the NOx emissions index of a specific combustor given a
    correlation from the ICAO EDB emissions testing.
    """

    def setup(self):
        self.add_input("T3", val=1600.0, desc="Burner inlet temperature at SLS", units="degK")
        self.add_input("P3", val=100.0, desc="Burner inlet pressure at SLS", units="lbf/inch**2")
        self.add_output("EINOx", val=10.0, desc="NOx emissions index at SLS")

        self.declare_partials("EINOx", ["T3", "P3"], method="cs")

    def compute(self, inputs, outputs):
        T3 = inputs["T3"]
        P3 = inputs["P3"]

        outputs["EINOx"] = P3 ** 0.4 * (
            (6.26e-8 * T3 ** 3) - (0.00117 * T3 ** 2) + (0.074 * T3) - 15.04
        )  # MIT NOx correlation for CFM56-5B3 engine


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
