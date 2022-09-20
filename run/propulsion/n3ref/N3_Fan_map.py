import numpy as np

from pycycle.maps.map_data import MapData


"""Python version of HBTF Fan map from NPSS"""
FanMap = MapData()

# Map design point values
FanMap.defaults = {}
FanMap.defaults['alphaMap'] = 0.0
FanMap.defaults['NcMap'] = 1.0
FanMap.defaults['PR'] = 1.39
FanMap.defaults['RlineMap'] = 2.00
FanMap.RlineStall = 1.0

FanMap.alphaMap = np.array([0.000, 5.000])
FanMap.NcMap = np.array([0.500, 0.600, 0.700, 0.800, 0.900, 0.950, 1.000, 1.050, 1.100])
FanMap.RlineMap = np.array([1.000, 1.250, 1.500, 1.750, 2.000, 2.250, 2.500, 2.750, 3.000, 3.200, 3.250, 3.500]) 

FanMap.WcMap = np.array([[[1104.3380, 1247.2229, 1384.7280, 1514.7731, 1635.3656, 1742.0557, 1829.9553, 1866.5565, 1866.5565, 1866.5565, 1866.5565, 1866.5565],
  [1369.0782, 1519.3645, 1661.7891, 1793.9161, 1913.4485, 2012.9872, 2084.3818, 2125.0225, 2126.4814, 2126.4814, 2126.4814, 2126.4814],
  [1644.4929, 1791.9211, 1929.5121, 2054.8865, 2165.8337, 2252.1497, 2302.7083, 2315.6926, 2315.6926, 2315.6926, 2315.6926, 2315.6926],
  [1919.7177, 2054.9785, 2179.5896, 2291.5830, 2389.1497, 2460.2048, 2491.7344, 2493.1619, 2493.1619, 2493.1619, 2493.1619, 2493.1619],
  [2191.8401, 2304.7476, 2407.9941, 2500.2908, 2580.4556, 2636.5833, 2655.9902, 2655.9902, 2655.9902, 2655.9902, 2655.9902, 2655.9902],
  [2324.3926, 2422.7034, 2512.5652, 2593.0525, 2663.3154, 2712.5415, 2729.2681, 2729.2681, 2729.2681, 2729.2681, 2729.2681, 2729.2681],
  [2446.8806, 2530.6360, 2607.3428, 2676.3726, 2737.1450, 2780.9336, 2798.5239, 2798.8225, 2798.8225, 2798.8225, 2798.8225, 2798.8225],
  [2564.0947, 2632.0720, 2694.6523, 2751.4587, 2802.1399, 2840.2256, 2858.9136, 2860.8069, 2860.8069, 2860.8069, 2860.8069, 2860.8069],
  [2675.1184, 2726.3958, 2774.0383, 2817.8599, 2857.6853, 2889.2947, 2908.2825, 2914.2979, 2914.2979, 2914.2979, 2914.2979, 2914.2979]],
  [[ 883.6700, 1019.1972, 1150.5194, 1275.3372, 1391.4280, 1494.5356, 1580.0111, 1639.0737, 1639.0737, 1639.0737, 1639.0737, 1639.0737],
  [1103.0485, 1250.1055, 1390.6215, 1521.7380, 1640.7301, 1740.2568, 1812.2133, 1853.6100, 1860.6985, 1860.6985, 1860.6985, 1860.6985],
  [1336.1071, 1487.0300, 1629.0968, 1759.2498, 1874.6134, 1964.5132, 2017.2321, 2030.5022, 2030.5022, 2030.5022, 2030.5022, 2030.5022],
  [1576.8409, 1724.1096, 1860.7565, 1983.9010, 2090.8706, 2168.0376, 2200.6162, 2201.4871, 2201.4871, 2201.4871, 2201.4871, 2201.4871],
  [1821.3027, 1957.1122, 2081.6206, 2192.4666, 2287.4802, 2351.2808, 2368.0996, 2368.0996, 2368.0996, 2368.0996, 2368.0996, 2368.0996],
  [1944.5967, 2071.4778, 2187.2837, 2290.0161, 2377.8403, 2435.2661, 2447.6707, 2447.6707, 2447.6707, 2447.6707, 2447.6707, 2447.6707],
  [2068.5525, 2184.2493, 2289.5361, 2382.8152, 2462.6226, 2514.0239, 2523.8010, 2523.8010, 2523.8010, 2523.8010, 2523.8010, 2523.8010],
  [2188.6860, 2292.2751, 2386.4270, 2469.9207, 2541.6360, 2588.0962, 2597.1982, 2597.1982, 2597.1982, 2597.1982, 2597.1982, 2597.1982],
  [2303.5840, 2394.5933, 2477.3633, 2551.0061, 2614.6997, 2657.1194, 2667.1841, 2667.1841, 2667.1841, 2667.1841, 2667.1841, 2667.1841]]])

FanMap.effMap = np.array([[[0.6585, 0.7756, 0.8552, 0.8821, 0.8316, 0.6642, 0.3039, 0.0010, 0.0010, 0.0010, 0.0010, 0.0010],
  [0.7145, 0.8102, 0.8747, 0.8970, 0.8603, 0.7400, 0.4938, 0.0386, 0.0005, 0.0005, 0.0005, 0.0005],
  [0.7692, 0.8439, 0.8942, 0.9128, 0.8896, 0.8113, 0.6571, 0.4108, 0.1588, 0.0046, 0.0046, 0.0046],
  [0.8178, 0.8734, 0.9113, 0.9273, 0.9159, 0.8710, 0.7838, 0.6910, 0.5830, 0.4661, 0.4314, 0.2151],
  [0.8588, 0.8965, 0.9227, 0.9356, 0.9327, 0.9118, 0.8700, 0.8384, 0.7979, 0.7568, 0.7451, 0.6766],
  [0.8749, 0.9042, 0.9250, 0.9359, 0.9357, 0.9232, 0.8968, 0.8791, 0.8568, 0.8345, 0.8282, 0.7920],
  [0.8857, 0.9081, 0.9242, 0.9334, 0.9350, 0.9283, 0.9127, 0.9016, 0.8903, 0.8791, 0.8759, 0.8579],
  [0.8926, 0.9086, 0.9204, 0.9278, 0.9302, 0.9275, 0.9195, 0.9118, 0.9072, 0.9026, 0.9012, 0.8936],
  [0.8945, 0.9049, 0.9128, 0.9182, 0.9207, 0.9204, 0.9172, 0.9109, 0.9099, 0.9088, 0.9085, 0.9063]],
  [[0.5665, 0.7209, 0.8444, 0.9236, 0.9354, 0.8420, 0.5653, 0.0165, 0.0165, 0.0165, 0.0165, 0.0165],
  [0.6269, 0.7585, 0.8631, 0.9301, 0.9428, 0.8756, 0.6810, 0.2552, 0.0013, 0.0013, 0.0013, 0.0013],
  [0.6849, 0.7935, 0.8792, 0.9345, 0.9479, 0.9035, 0.7748, 0.5410, 0.3281, 0.0381, 0.0069, 0.0069],
  [0.7365, 0.8231, 0.8912, 0.9358, 0.9495, 0.9234, 0.8439, 0.7726, 0.6863, 0.5806, 0.5470, 0.3126],
  [0.7802, 0.8461, 0.8979, 0.9324, 0.9457, 0.9331, 0.8953, 0.8751, 0.8429, 0.8052, 0.7936, 0.7198],
  [0.7988, 0.8546, 0.8985, 0.9282, 0.9407, 0.9332, 0.9106, 0.8999, 0.8820, 0.8610, 0.8546, 0.8143],
  [0.8147, 0.8607, 0.8969, 0.9218, 0.9332, 0.9295, 0.9165, 0.9122, 0.9037, 0.8931, 0.8899, 0.8694],
  [0.8266, 0.8637, 0.8931, 0.9136, 0.9239, 0.9230, 0.9153, 0.9150, 0.9123, 0.9080, 0.9066, 0.8974],
  [0.8343, 0.8637, 0.8870, 0.9036, 0.9127, 0.9136, 0.9086, 0.9106, 0.9113, 0.9106, 0.9103, 0.9075]]])

FanMap.PRmap = np.array([[[1.0760, 1.0811, 1.0799, 1.0724, 1.0588, 1.0394, 1.0146, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
  [1.1171, 1.1214, 1.1186, 1.1085, 1.0914, 1.0676, 1.0379, 1.0024, 1.0000, 1.0000, 1.0000, 1.0000],
  [1.1703, 1.1730, 1.1682, 1.1559, 1.1363, 1.1100, 1.0776, 1.0414, 1.0133, 1.0003, 1.0003, 1.0003],
  [1.2381, 1.2386, 1.2320, 1.2185, 1.1981, 1.1715, 1.1390, 1.1094, 1.0814, 1.0582, 1.0523, 1.0222],
  [1.3234, 1.3215, 1.3138, 1.3004, 1.2814, 1.2572, 1.2282, 1.2039, 1.1789, 1.1584, 1.1531, 1.1265],
  [1.3734, 1.3703, 1.3624, 1.3497, 1.3323, 1.3104, 1.2845, 1.2628, 1.2404, 1.2221, 1.2174, 1.1937],
  [1.4288, 1.4250, 1.4172, 1.4055, 1.3900, 1.3709, 1.3484, 1.3284, 1.3090, 1.2931, 1.2891, 1.2686],
  [1.4898, 1.4854, 1.4780, 1.4677, 1.4545, 1.4386, 1.4201, 1.4022, 1.3862, 1.3731, 1.3698, 1.3530],
  [1.5553, 1.5507, 1.5441, 1.5356, 1.5251, 1.5128, 1.4988, 1.4831, 1.4708, 1.4608, 1.4583, 1.4455]],
  [[1.0677, 1.0776, 1.0807, 1.0769, 1.0662, 1.0490, 1.0259, 1.0006, 1.0006, 1.0006, 1.0006, 1.0006],
  [1.1061, 1.1167, 1.1192, 1.1135, 1.0996, 1.0781, 1.0496, 1.0144, 1.0001, 1.0001, 1.0001, 1.0001],
  [1.1560, 1.1661, 1.1673, 1.1595, 1.1430, 1.1181, 1.0858, 1.0492, 1.0236, 1.0022, 1.0004, 1.0004],
  [1.2188, 1.2274, 1.2269, 1.2173, 1.1989, 1.1722, 1.1379, 1.1090, 1.0818, 1.0591, 1.0534, 1.0239],
  [1.2966, 1.3028, 1.3005, 1.2898, 1.2707, 1.2439, 1.2118, 1.1859, 1.1590, 1.1368, 1.1311, 1.1022],
  [1.3418, 1.3465, 1.3433, 1.3322, 1.3134, 1.2875, 1.2576, 1.2326, 1.2067, 1.1853, 1.1799, 1.1522],
  [1.3913, 1.3943, 1.3902, 1.3790, 1.3610, 1.3364, 1.3089, 1.2854, 1.2610, 1.2410, 1.2359, 1.2100],
  [1.4454, 1.4468, 1.4420, 1.4310, 1.4140, 1.3912, 1.3658, 1.3441, 1.3218, 1.3034, 1.2987, 1.2751],
  [1.5044, 1.5044, 1.4991, 1.4886, 1.4728, 1.4523, 1.4287, 1.4091, 1.3890, 1.3725, 1.3683, 1.3471]]])

#FanMap.Nc_data, FanMap.alpha_data, FanMap.Rline_data = np.meshgrid(FanMap.Nc_vals, FanMap.alpha_vals, FanMap.Rline_vals, sparse=False)
FanMap.Npts = FanMap.NcMap.size

FanMap.units = {}
FanMap.units['NcMap'] = 'rpm'
FanMap.units['WcMap'] = 'lbm/s'


# format for new regular grid interpolator:

FanMap.param_data = []
FanMap.output_data = []

FanMap.param_data.append({'name': 'alphaMap', 'values': FanMap.alphaMap,
                          'default': 0, 'units': None})
FanMap.param_data.append({'name': 'NcMap', 'values': FanMap.NcMap,
                          'default': 1.0, 'units': 'rpm'})
FanMap.param_data.append({'name': 'RlineMap', 'values': FanMap.RlineMap,
                          'default': 2.0, 'units': None})

FanMap.output_data.append({'name': 'WcMap', 'values': FanMap.WcMap,
                           'default': np.mean(FanMap.WcMap), 'units': 'lbm/s'})
FanMap.output_data.append({'name': 'effMap', 'values': FanMap.effMap,
                           'default': np.mean(FanMap.effMap), 'units': None})
FanMap.output_data.append({'name': 'PRmap', 'values': FanMap.PRmap,
                           'default': 1.39, 'units': None})