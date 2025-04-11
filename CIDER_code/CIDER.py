import numpy as np
from scipy.special import erfc
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from Toolbox import global_mean


def AOD_from_injection(param_AOD, injection):
    # Calculates AOD from SO2 injection at one latitude
    # param_AOD_all: AOD parameters, for all injection latitudes
    # injection: single latitude injection timeseries (monthly)
    beta, alpha, gamma = param_AOD
    injection = np.array(injection)
    AOD_emulated = np.zeros(len(injection))

    if np.mean(injection) == 0:
        return AOD_emulated

    for k in range(len(AOD_emulated)):
        for j in range(k + 1):
            AOD_emulated[k] += (
                impulse_firstOdiff_wGammaInj(beta, alpha, gamma, injection[k - j], j +1)
                * injection[k - j]
            )

    return AOD_emulated


def AOD_pattern_from_all_injections(param_AOD_all, injections, patterns):
    # Calculates AOD pattern from all injections.
    # param_AOD_all: AOD parameters, for all injection latitudes
    # injections: all injection timeseries (monthly), formatted as a matrix
    # patterns: all AOD patterns, formatted as a matrix
    size_injections = injections.shape
    injection_count = size_injections[1]
    injection_length = size_injections[0]
    size_patterns = patterns.shape

    # Initialize the total AOD pattern array
    total_AOD_pattern = np.zeros((size_patterns[0], size_patterns[1], injection_length))

    for i in range(injection_count):
        injection = injections[:, i]
        param_AOD = param_AOD_all[i, :]
        pattern = patterns[:, :, i]

        # Calculate AOD emulated for the current injection
        AOD_emulated = AOD_from_injection(param_AOD, injection)

        # Scale the pattern with the AOD emulated
        scaled_pattern = pattern_scale(AOD_emulated, pattern)

        # Add the scaled pattern to the total
        total_AOD_pattern += scaled_pattern

    return total_AOD_pattern


def get_pattern(data, steady_length):
    # Pull the 2-D pattern from data, averaging over last steady_length entries
    # Extract the data to average over the steady-state period
    data_to_average = data[:, :, -steady_length:]

    # Compute the mean along the third dimension
    mean_data = np.mean(data_to_average, axis=2)

    # Normalize by the global mean
    pattern = mean_data / global_mean(mean_data)
    return pattern

def pattern_from_1_injection(injection, param_AOD, param_climate, pattern_to_scale):
    # Emulate response pattern from 1 latitude of SAI injection
    # injection: single latitude injection timeseries (monthly)
    # param_AOD: inj->AOD parameters
    # param_climate: AOD->climate parameters
    # pattern_to_scale: climate pattern from that latitude

    # Emulate the AOD from the injection
    AOD_emulated = AOD_from_injection(param_AOD, injection)

    # Compute the emulated response from the AOD
    emulated_response = response_from_1_forcing(param_climate, AOD_emulated)

    # Scale the pattern with the emulated response
    response_pattern = pattern_scale(emulated_response, pattern_to_scale)
    return response_pattern



def pattern_from_all_injections(all_injection, all_param_AOD, all_param_climate, all_pattern_to_scale):
    # Emulate response pattern from all latitudes of SAI injection
    # all_injection: all injection timeseries (monthly), formatted as a matrix
    # all_param_AOD: AOD parameters, for all injection latitudes
    # all_param_climate: climate parameters, for all injection latitudes
    # all_pattern_to_scale: all climate patterns, formatted as a matrix
    
    injection_length, injection_count = all_injection.shape
    pattern_dim1, pattern_dim2, _ = all_pattern_to_scale.shape

    # Initialize the total pattern array
    total_pattern = np.zeros((pattern_dim1, pattern_dim2, injection_length))

    for i in range(injection_count):
        injection = all_injection[:, i]
        if np.mean(injection) != 0:
            param_AOD = all_param_AOD[i, :]
            param_climate = all_param_climate[i, :]
            pattern_to_scale = all_pattern_to_scale[:, :, i]

            # Compute the response pattern for the current injection
            response_pattern = pattern_from_1_injection(
                injection, param_AOD, param_climate, pattern_to_scale
            )

            # Add the response pattern to the total
            total_pattern += response_pattern

    return total_pattern


def pattern_from_all_injections_and_CO2(all_injection_and_CO2, all_param_AOD, all_param_climate, all_pattern_to_scale):
    # Emulate response pattern from all latitudes of SAI injection and CO2
    # all_injection_and_CO2: all injection timeseries and CO2 forcing (monthly), forcing as last column
    # all_param_AOD: AOD parameters, for all injection latitudes
    # all_param_climate: climate parameters, for all injection latitudes and CO2 (which is last)
    # all_pattern_to_scale: all climate patterns from injections and  and CO2 (which is last)
    
    total_pattern_inj = pattern_from_all_injections(
        all_injection_and_CO2[:, :-1], all_param_AOD, all_param_climate[:-1, :], all_pattern_to_scale
    )
    pattern_CO2 = pattern_scale(
        response_from_1_forcing(all_param_climate[-1, :], all_injection_and_CO2[:, -1]),
        all_pattern_to_scale[:, :, -1]
    )
    total_pattern = total_pattern_inj + pattern_CO2

    return total_pattern


def pattern_scale(mean_response, pattern_to_scale):
    # Scale pattern_to_scale by mean_response
    pattern_to_scale = pattern_to_scale[:,:,np.newaxis]
    pattern = pattern_to_scale * mean_response.reshape(1, 1, -1)
    return pattern


def response_from_1_forcing(params, forcing):
    # Emulated response for climate from a single forcing
    impulse_p_SAI = {'tau': params[0], 'mu': params[1]}
    emulated_response = np.zeros(len(forcing))

    for k in range(len(emulated_response)):
        for j in range(k + 1):
            emulated_response[k] += impulse_semiInfDiff(j+1, impulse_p_SAI) * forcing[k - j]

    return emulated_response


def response_from_all_injections_and_CO2(all_injection_and_CO2, all_param_AOD, all_param_climate):
    # Emulated response for climate from all forcings (GHG last)
    injection_count = all_injection_and_CO2.shape[1] - 1
    total_response = response_from_1_forcing(all_param_climate[-1, :], all_injection_and_CO2[:, -1])

    for i in range(injection_count):
        injection = all_injection_and_CO2[:, i]
        param_AOD = all_param_AOD[i, :]
        param_climate = all_param_climate[i, :]
        AOD = AOD_from_injection(param_AOD, injection)
        total_response += response_from_1_forcing(param_climate, AOD)

    return total_response


def train_AOD_params(all_step_injections, all_step_responses, all_feedback_injections, all_feedback_responses, weightings):
    step_injections_count = all_step_injections.shape[1]
    all_params = np.zeros((step_injections_count, 3))

    for i in range(step_injections_count):
        step_injection = all_step_injections[:, i]
        feedback_injection = all_feedback_injections[:, i]
        simulated_step_response = all_step_responses[:, i]
        simulated_feedback_response = all_feedback_responses[:, i]

        def step_error(params):
            return np.sum((simulated_step_response - AOD_from_injection(params, step_injection))**2) / len(simulated_step_response)

        def feedback_error(params):
            return np.sum((simulated_feedback_response - AOD_from_injection(params, feedback_injection))**2) / len(simulated_feedback_response)

        def emulator_error(params):
            return weightings[0] * step_error(params) + weightings[1] * feedback_error(params)

        x0 = np.array([0.02, 0.1, 0.02])

        # Use minimize to optimize the parameters
        result = minimize(emulator_error, x0, bounds=[(0, 0.1), (0, 0.2), (0, 0.1)])
        optimal_params = result.x

        all_params[i, :] = optimal_params

    return all_params


def train_climate_params(all_forcings, all_responses, x0=None, lb=None, ub=None):
    forcings_count = all_forcings.shape[1]
    all_params = np.zeros((forcings_count, 2))

    if x0 is None:
        x0 = np.array([30, 0])
    if lb is None:
        lb = np.array([0, -1])
    if ub is None:
        ub = np.array([500, 1])

    for i in range(forcings_count):
        forcing = all_forcings[:, i]
        simulated_response = all_responses[:, i]

        def emulator_error(params):
            return np.sum((simulated_response - response_from_1_forcing(params, forcing))**2) / len(simulated_response)

        result = minimize(emulator_error, x0, bounds=[(lb[0], ub[0]), (lb[1], ub[1])])
        optimal_params = result.x

        all_params[i, :] = optimal_params

    return all_params


def impulse_firstOdiff_wGammaInj(beta, alpha, gamma, q, t):
    # Impulse response for AOD
    ydata = beta*np.exp(-(alpha+gamma*q)*t)
    return ydata


def impulse_semiInfDiff(t, impulse_p):
    # Impulse response for climate
    if t == 0:
        t = np.finfo(float).tiny
    mu = impulse_p['mu']
    tau = impulse_p['tau']
    h = mu * (1 / np.sqrt(np.pi * t / tau) - np.exp(t / tau) * erfc(np.sqrt(t / tau)))
    if np.isinf(np.exp(t / tau) * erfc(np.sqrt(t / tau))) or np.isnan(np.exp(t / tau) * erfc(np.sqrt(t / tau))):
        h = mu * (1 / np.sqrt(np.pi * t / tau) - 2 / np.sqrt(np.pi) * (np.sqrt(t / tau) + np.sqrt(np.sqrt(t / tau)**2 + 2))**(-1))
    return h