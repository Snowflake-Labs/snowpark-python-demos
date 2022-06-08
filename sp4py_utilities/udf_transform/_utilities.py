import numpy as np

def _check_fitted(encoder_type, encoder):

    if not "encoder" in encoder:
        raise TypeError("The provided encoder is not a valid encoder.")

    if not encoder["encoder"] == encoder_type:
        raise TypeError(f"The encoder is not a {encoder_type}.")

    if not "fitted_values" in encoder:
        raise TypeError("The provided encoder is not fitted.")


def _verify_input(data, nbr_features):
    if not isinstance(data, np.ndarray):
        if not any(isinstance(i, list) for i in data):
            found_cols = len(data)
        else:
            found_cols = len(data[0])
    else:
        _, found_cols = data.shape

    if nbr_features != found_cols:
        raise ValueError("X does not have required number of columns.")