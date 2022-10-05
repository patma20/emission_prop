# --- Python 3.8 ---
"""
@File    :   constrained_balance.py
@Time    :   2020/11/18
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

h = 1000.0


def sig(x):
    return 1 / (1 + np.exp(-h * x))


class FAR_Balance_1(om.ImplicitComponent):
    """
    Implicit component which varies the temperature to converge the corrected fan speed to 100% as long as
    the temperature stays below T_requested
    """

    def setup(self):
        self.add_input("T4", val=3500, units="degR")
        self.add_input("T4max", val=3660, units="degR")
        self.add_input("T3", val=1600, units="degR")
        self.add_input("T3max", val=1750, units="degR")
        self.add_input("NcMapVal", val=0.98, units="rpm")
        self.add_input("NcMapTgt", val=1.0, units="rpm")

        self.add_output("FAR", val=0.034, upper=0.06, lower=0.01, units=None)

        self.declare_partials("FAR", ["T3", "T4", "NcMapVal", "T4max"])  # , method="fd")

    def apply_nonlinear(self, inputs, outputs, residuals):

        T4 = inputs["T4"]
        T4max = inputs["T4max"]

        T3 = inputs["T3"]
        T3max = inputs["T3max"]

        NcMapVal = inputs["NcMapVal"]
        NcMapTgt = inputs["NcMapTgt"]

        if T4 >= T4max:
            residuals["FAR"] = (T4 - T4max) / T4max

        elif NcMapVal >= NcMapTgt:
            residuals["FAR"] = (NcMapVal - NcMapTgt) / NcMapTgt

        else:
            residuals["FAR"] = (T3 - T3max) / T3max

    def linearize(self, inputs, outputs, partials):
        T4 = inputs["T4"]
        T4max = inputs["T4max"]

        T3max = inputs["T3max"]

        NcMapVal = inputs["NcMapVal"]
        NcMapTgt = inputs["NcMapTgt"]

        if T4 >= T4max:
            partials["FAR", "T4"] = 1 / T4max
            partials["FAR", "T4max"] = -T4 / T4max ** 2
            partials["FAR", "T3"] = 0.0
            partials["FAR", "NcMapVal"] = 0.0

        elif NcMapVal >= NcMapTgt:
            partials["FAR", "T4"] = 0.0
            partials["FAR", "T4max"] = 0.0
            partials["FAR", "T3"] = 0.0
            partials["FAR", "NcMapVal"] = 1 / NcMapTgt

        else:
            partials["FAR", "T4"] = 0.0
            partials["FAR", "T4max"] = 0.0
            partials["FAR", "T3"] = 1 / T3max
            partials["FAR", "NcMapVal"] = 0.0


class FAR_Balance_2(om.ImplicitComponent):
    def setup(self):

        self.add_input("T4", val=3500, units="degR")
        self.add_input("T4max", val=3660, units="degR")
        self.add_input("T3", val=1600, units="degR")
        self.add_input("T3max", val=1750, units="degR")
        self.add_input("NcMapVal", val=0.98, units="rpm")
        self.add_input("NcMapTgt", val=1.0, units="rpm")

        self.add_output("FAR", val=0.034, upper=0.06, lower=0.01, units=None)

        self.declare_partials("FAR", ["T3", "T4", "NcMapVal", "T4max"], method="fd")

    def apply_nonlinear(self, inputs, outputs, residuals):

        T4 = inputs["T4"]
        T4max = inputs["T4max"]

        T3 = inputs["T3"]
        T3max = inputs["T3max"]

        NcMapVal = inputs["NcMapVal"]
        NcMapTgt = inputs["NcMapTgt"]

        R_T3 = (T3 - T3max) / T3max

        R_T4 = (T4 - T4max) / T4max

        R_Nc = (NcMapVal - NcMapTgt) / NcMapTgt

        if T4 >= T4max:
            residuals["FAR"] = R_T4 + R_T3 * sig(R_T3) + R_Nc * sig(R_Nc)

        elif NcMapVal >= NcMapTgt:
            residuals["FAR"] = R_Nc + R_T3 * sig(R_T3) + R_T4 * sig(R_T4)

        else:
            residuals["FAR"] = R_T3 + R_T4 * sig(R_T4) + R_Nc * sig(R_Nc)

    # def linearize(self, inputs, outputs, partials):
    #     T4 = inputs["T4"]
    #     T4max = inputs["T4max"]

    #     T3 = inputs["T3"]
    #     T3max = inputs["T3max"]

    #     NcMapVal = inputs["NcMapVal"]
    #     NcMapTgt = inputs["NcMapTgt"]

    #     if T4 >= T4max:
    #         partials["FAR", "T4"] = 1 / T4max

    #         partials["FAR", "T4max"] = -T4 / T4max ** 2

    #         partials["FAR", "T3"] = (T3max * (np.exp(h * (T3 - T3max) / T3max) + 1) + h * (T3 - T3max)) / (
    #             4 * T3max ** 2 * np.cosh(h * (T3 - T3max) / (2 * T3max)) ** 2
    #         )

    #         partials["FAR", "NcMapVal"] = (
    #             NcMapTgt * (np.exp(h * (NcMapTgt - NcMapVal) / NcMapTgt) + 1)
    #             - h * (NcMapTgt - NcMapVal) * np.exp(h * (NcMapTgt - NcMapVal) / NcMapTgt)
    #         ) / (NcMapTgt ** 2 * (np.exp(h * (NcMapTgt - NcMapVal) / NcMapTgt) + 1) ** 2)

    #     elif NcMapVal >= NcMapTgt:
    #         partials["FAR", "T4"] = (T4max * (np.exp(h * (T4 - T4max) / T4max) + 1) + h * (T4 - T4max)) / (
    #             4 * T4max ** 2 * np.cosh(h * (T4 - T4max) / (2 * T4max)) ** 2
    #         )

    #         partials["FAR", "T4max"] = -(
    #             T4 * h * (T4 - T4max)
    #             + T4max ** 2 * (np.exp(h * (T4 - T4max) / T4max) + 1)
    #             + T4max * (T4 - T4max) * (np.exp(h * (T4 - T4max) / T4max) + 1)
    #         ) / (4 * T4max ** 3 * np.cosh(h * (T4 - T4max) / (2 * T4max)) ** 2)

    #         partials["FAR", "T3"] = (T3max * (np.exp(h * (T3 - T3max) / T3max) + 1) + h * (T3 - T3max)) / (
    #             4 * T3max ** 2 * np.cosh(h * (T3 - T3max) / (2 * T3max)) ** 2
    #         )

    #         partials["FAR", "NcMapVal"] = 1 / NcMapTgt

    #     else:
    #         partials["FAR", "T4"] = 1 / (T4max * (1 + np.exp(-h * (T4 - T4max) / T4max))) + h * (T4 - T4max) * np.exp(
    #             -h * (T4 - T4max) / T4max
    #         ) / (T4max ** 2 * (1 + np.exp(-h * (T4 - T4max) / T4max)) ** 2)

    #         partials["FAR", "T4max"] = (
    #             -1 / (T4max * (1 + np.exp(-h * (T4 - T4max) / T4max)))
    #             - (T4 - T4max)
    #             * (h / T4max + h * (T4 - T4max) / T4max ** 2)
    #             * np.exp(-h * (T4 - T4max) / T4max)
    #             / (T4max * (1 + np.exp(-h * (T4 - T4max) / T4max)) ** 2)
    #             - (T4 - T4max) / (T4max ** 2 * (1 + np.exp(-h * (T4 - T4max) / T4max)))
    #         )
    #         partials["FAR", "T3"] = 1 / T3max
    #         partials["FAR", "NcMapVal"] = (
    #             NcMapTgt * (np.exp(h * (NcMapTgt - NcMapVal) / NcMapTgt) + 1)
    #             - h * (NcMapTgt - NcMapVal) * np.exp(h * (NcMapTgt - NcMapVal) / NcMapTgt)
    #         ) / (NcMapTgt ** 2 * (np.exp(h * (NcMapTgt - NcMapVal) / NcMapTgt) + 1) ** 2)


if __name__ == "__main__":
    prob = om.Problem()
    prob.model = FAR_Balance_2()
    prob.setup()
    prob.run_model()
    prob.check_partials(method="cs", compact_print=True)
