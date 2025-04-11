import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from Toolbox import global_mean, lat_band_mean, repeat_elements

mat = scipy.io.loadmat('parameters/CESM_params.mat')
# mat = scipy.io.loadmat('new_UKESM_params.mat')
param_AOD_all = np.array(mat['param_AOD_all'])
param_T_all = np.array(mat['param_T_all'])
param_P_all = np.array(mat['param_P_all'])
all_T_patterns_scaled = np.array(mat['pattern_T_all'])
T_base_pattern = np.array(mat['pattern_T_base'])
CO2_file = scipy.io.loadmat('CO2_concentrations.mat')
CO2_conc_SSP245 = np.array(CO2_file['CO2_SSP245'])
CO2levels_2035_2100_ssp245 = CO2_conc_SSP245[6+14:85].reshape(-1, 1);
CO2levels_2035_2069_ssp245 = CO2_conc_SSP245[6+14:55].reshape(-1, 1);
CO2_ref = CO2_conc_SSP245[5+14];
CO2_forcing_SSP245 = 5.35*np.log((CO2levels_2035_2100_ssp245)/CO2_ref);
CO2_forcing_SSP245_month = repeat_elements(CO2_forcing_SSP245,12);
