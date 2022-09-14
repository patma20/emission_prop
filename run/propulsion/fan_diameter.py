# --- Python 3.8 ---
"""
Computes the fan diameter
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


class FanDiameter(om.ExplicitComponent):
    def setup(self):
        self.add_input("area", val=7000.0, desc="Fan face area", units="inch**2")
        self.add_input("hub_tip", val=0.3, desc="Hub to tip ratio", units=None)

        self.add_output("fan_dia", val=55.0, desc="Fan diameter", units="inch")

        self.declare_partials("*", "*")

    def compute(self, inputs, outputs):
        area = inputs["area"]
        hub_tip = inputs["hub_tip"]

        outputs["fan_dia"] = 2.0 * (area / (np.pi * (1.0 - hub_tip ** 2.0))) ** 0.5

    def compute_partials(self, inputs, partials):
        area = inputs["area"]
        hub_tip = inputs["hub_tip"]

        partials["fan_dia", "area"] = np.pi ** (-0.5) * (area / (1.0 - hub_tip ** 2.0)) ** 0.5 / area
        partials["fan_dia", "hub_tip"] = (
            2.0 * np.pi ** (-0.5) * hub_tip * (area / (1.0 - hub_tip ** 2.0)) ** 0.5 / (1.0 - hub_tip ** 2.0)
        )
