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
        self.add_input("P3", val=14.7, units="lfb/inch**2")
        self.add_input("T3", val=518.67, units="R")
        self.add_input("T4", val=518.67, units="R")

        self.add_output("EINOx", val=1.0, desc="NOx emissions index")
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
