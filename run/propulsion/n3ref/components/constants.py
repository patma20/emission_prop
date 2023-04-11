#!/usr/bin/env python
"""
@File    :   constants.py
@Time    :   2022/02/10
@Desc    :   None
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================

# Mobil Jet Oil II (https://www.exxonmobil.com/en-us/aviation/pds/gl-xx-mobil-jet-oil-ii)
rho_oil = 1003.5  # kg/m^3
nu_oil = 5.1 / (1000 ** 2)  # m^2 / s
mu_oil = nu_oil * rho_oil

# Mobil Jet Oil 254 (https://www.exxonmobil.com/en/aviation/products-and-services/products/mobil-jet-oil-254)
# rho_oil = 1004.4  # kg/m^3
# nu_oil = 5.3 / (1000 ** 2)  # m^2 / s
# mu_oil = nu_oil * rho_oil

k_oil = 0.140  # W/m/K (Bair)
cp_oil = 2500  # J/kg/K

# HX Materials (Aluminum)
rho_al = 2700  # kg/m^3
k_al = 190  # W/kg/K
cp_al = 921  # J/kg/K

LHV_h2 = 51591  # Lower Heating Value of H2 in BTU/lb
LHV_JetA = 18564  # Lower Heating Value of JetA in BTU/lb
