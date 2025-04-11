import numpy as np

def global_mean(M):
    latitudes = np.arange(-90, 90, 180/M.shape[1])
    ww = np.cos(np.deg2rad(latitudes))
    if M.ndim == 2:
        # aa = M[:, :].T * ww
        av = np.mean(np.sum(M[:, :] * ww, axis=1) / np.sum(ww))
        return av

    if M.ndim == 3:
        av = np.zeros((M.shape[2], 1))
        for i in range(M.shape[2]):
            # aa = M[:, :,i].T * ww
            av[i] = np.mean(np.sum(M[:, :, i] * ww, axis=1) / np.sum(ww))
        return av


def lat_band_mean(M, lower_bound, upper_bound):
    latitudes = np.arange(-90, 90, 180/M.shape[1])
    ww = np.cos(np.deg2rad(latitudes))
    is_in_bounds = (latitudes <= upper_bound) & (latitudes >= lower_bound)
    if M.ndim == 2:
        av = np.mean(np.sum(M[:, is_in_bounds] * ww[is_in_bounds], axis=1) / np.sum(ww[is_in_bounds]))
        return av

    if M.ndim == 3:
        av = np.zeros((M.shape[2], 1))
        for i in range(M.shape[2]):
            av[i] = np.mean(np.sum(M[:, is_in_bounds, i] * ww[is_in_bounds], axis=1) / np.sum(ww[is_in_bounds]))
        return av


def repeat_elements(input_array, N):
    # Convert input_array to a NumPy array and ensure it's a column vector
    input_array = np.array(input_array).reshape(-1, 1)
    # Create a matrix of shape (N, len(input_array)) where each row is input_array
    matrix_for_reshaping = np.tile(input_array.T, (N, 1)).T
    # Flatten the resulting matrix to get a 1D output array
    output_array = matrix_for_reshaping.flatten()
    return output_array

def stack_and_zoh_injections(inj_tuple,zero_order_hold_length):
    injection_all = np.vstack(inj_tuple).T
    injection_all_zoh = np.zeros((injection_all.shape[0]*zero_order_hold_length,7))
    for i in range(injection_all.shape[1]):
        injection_all_zoh[:,i] = repeat_elements(injection_all[:,i], zero_order_hold_length)
    return injection_all_zoh/12

