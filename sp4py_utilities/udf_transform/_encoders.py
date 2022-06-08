import numpy as np

from ._utilities import _check_fitted, _verify_input

DEFAULT_UNKNOWN_VALUE = -999999


def udf_onehot_transform(X, encoder):
    """
    Transform input columns of a list or numpy array

    :param X: A list or Numpy Array
    :param encoder: A encoder returned by OneHotEncoder get_udf_encoder function
    :return: A transformed Numpy array
    """
    _check_fitted('OneHotEncoder', encoder)

    n_features = encoder['nbr_features']
    _verify_input(X, n_features)

    fitted_values = encoder["fitted_values"]
    # We need to convert it to a list of list, but be able to go back
    nested_array = True
    if not any(isinstance(i, list) for i in X):
        X = [X]
        # We need to convert it to a list of list, but be able to go back
        nested_array = False

    # Need to create an array with the total number of categories in fitted values
    add_unk = 0
    if encoder["handle_unknown"] == 'keep':
        add_unk = 1

    needed_cols_cat = [len(fitted_values[col]) + add_unk for col in encoder["input_features"]]

    X_ret = np.zeros((len(X), sum(needed_cols_cat)), dtype=int)
    # It is not the most optimal way of doing this, but it gets the work done...
    for idx, row in enumerate(X):
        start_idx = 0
        for i in range(n_features):
            uniques = fitted_values[encoder['input_features'][i]]
            set_flag = True
            # Check if we have an unknown value
            if not str(row[i]) in uniques:
                # If keep then set 1 in the "unknown" column
                if encoder["handle_unknown"] == 'keep':
                    flag_idx = start_idx + needed_cols_cat[i] - 1
                # else we should ignore, but
                else:
                    set_flag = False
            else:
                flag_idx = np.searchsorted(uniques, row[i]) + start_idx
            if set_flag:
                X_ret[idx, flag_idx] = 1
            start_idx += needed_cols_cat[i]

    return X_ret if nested_array else X_ret[0]


def udf_onehot_inverse_transform(X, encoder):
    """
    Reverse the encoding.

    :param X: A list or Numpy Array that has been transformed using udf_onehot_transform
    :param encoder: A encoder returned by OrdinalEncoder get_udf_encoder function
    :return: A reversed Numpy array
    """
    _check_fitted('OneHotEncoder', encoder)

    fitted_values = encoder["fitted_values"]
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    nested_array = True
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
        # We need to convert it to a list of list, but be able to go back
        nested_array = False

    # We need to create a new array with the number of features used for transform
    n_cols = len(encoder['input_features'])
    n_rows, _ = X.shape

    # Return array is the original features for
    X_ret = np.empty((n_rows, n_cols), dtype=object)

    # input_cols has the correct order of values
    j = 0
    found_unknown = {}
    for i, col in enumerate(encoder['input_features']):
        cats = np.array(fitted_values[col])
        n_categories = len(cats)
        sub = X[:, j:j + n_categories]
        # Labels will have 0 and 1, 1
        labels = np.asarray(sub.argmax(axis=1)).flatten()
        X_ret[:, i] = np.array(cats)[labels]
        # Check for unknown values
        unknown = np.asarray(sub.sum(axis=1) == 0).flatten()
        if unknown.any():
            found_unknown[i] = unknown
        # If we have kept unknown categories we need to skip that column (last)
        if encoder['handle_unknown'] == "keep":
            n_categories += 1
        j += n_categories

    if found_unknown:
        if X_ret.dtype != object:
            X_ret = X_ret.astype(object)
        for idx, mask in found_unknown.items():
            X_ret[mask, idx] = None

    return X_ret if nested_array else X_ret[0]


def udf_ordinal_transform(X, encoder):
    """
    Transform

    :param X: A list or Numpy Array to be transformed
    :param encoder: A encoder returned by OrdinalEncoder get_udf_encoder function
    :return: A reversed Numpy array
    """
    _check_fitted('OrdinalEncoder', encoder)

    n_features = encoder['nbr_features']

    _verify_input(X, n_features)

    fitted_values = encoder["fitted_values"]

    # We need to convert it to a list of list, but be able to go back
    nested_array = True
    if not any(isinstance(i, list) for i in X):
        X = [X]
        nested_array = False

    X_columns = []
    X_temp = np.asarray(X)
    for i in range(n_features):
        Xi = X_temp[:, i]
        X_columns.append(Xi)

    X_list = X_columns
    # _check_X END

    # Index of matched value in fitted_values for same column, one row for each row in X
    X_int = np.zeros((len(X), n_features), dtype=int)
    X_mask = np.ones((len(X), n_features), dtype=bool)
    unknowns = False

    for i in range(n_features):
        Xi = X_list[i]
        uniques = fitted_values[encoder['input_features'][i]]
        transform_values = set(Xi)
        known_values = set(uniques)

        def is_valid(value):
            return (
                    value in known_values
                    or value is None
            )
        diff = transform_values - known_values
        if diff:
            # True if the value exists in known_values else false
            valid_mask = np.array([is_valid(value) for value in Xi])
        else:
            # All to true since nothing is missing
            valid_mask = np.ones(len(transform_values), dtype=bool)
        # If we have unknowns...
        if not np.all(valid_mask):
            unknowns = True
            X_mask[:, i] = valid_mask
            Xi[~valid_mask] = fitted_values[encoder['input_features'][i]][0]

        X_int[:, i] = np.searchsorted(fitted_values[encoder['input_features'][i]], Xi)

    X_trans = X_int.astype(np.int, copy=False)
    # Fix unknown values
    if encoder["handle_unknown"] == "use_encoded_value":
        X_trans[~X_mask] = encoder["unknown_value"]
        unknowns = False
    if unknowns:
        X_trans[~X_mask] = DEFAULT_UNKNOWN_VALUE #None

    return X_trans if nested_array else X_trans[0]


def udf_ordinal_inverse_transform(X, encoder):
    #
    _check_fitted('OrdinalEncoder', encoder)

    fitted_values = encoder["fitted_values"]

    if not isinstance(X, np.ndarray):
        X = np.array(X)

    nested_array = True
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
        # We need to convert it to a list of list, but be able to go back
        nested_array = False

    # We need to create a new array with the number of features used for transform
    n_cols = len(encoder['input_features'])
    n_rows, _ = X.shape

    # Return array is the original features for
    X_ret = np.empty((n_rows, n_cols), dtype=object)
    found_unknown = {}

    for i, col in enumerate(encoder['input_features']):
        labels = X[:, i].astype("int64", copy=False)
        cats = np.array(fitted_values[col])
        if encoder['handle_unknown'] == "use_encoded_value":
            unknown_labels = labels == encoder['unknown_value']
            #
            #X_ret[:, i] = cats[np.where(unknown_labels, 0, labels)]
            #found_unknown[i] = unknown_labels
        else:
            # If we have used ignore
            unknown_labels = labels == DEFAULT_UNKNOWN_VALUE
            # Need to check if there are
            #X_ret[:, i] = cats[labels]
        X_ret[:, i] = cats[np.where(unknown_labels, 0, labels)]
        found_unknown[i] = unknown_labels

    # insert None values for unknown values
    if found_unknown:
        X_ret = X_ret.astype(object, copy=False)

        for idx, mask in found_unknown.items():
            X_ret[mask, idx] = None

    return X_ret if nested_array else X_ret[0]


def udf_label_transform(Y, encoder):
    #
    _check_fitted('LabelEncoder', encoder)

    fitted_values = encoder["fitted_values"]
    nested_array = True
    if not any(isinstance(i, list) for i in Y):
        Y = [Y]
        nested_array = False

    Y_trans = np.searchsorted(fitted_values[encoder['input_features']], Y)

    return Y_trans if nested_array else Y_trans[0]


def udf_label_inverse_transform(Y, encoder):

    _check_fitted('LabelEncoder', encoder)

    fitted_values = encoder["fitted_values"]

    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)

    nested_array = True
    if Y.ndim == 1:
        Y = np.expand_dims(Y, axis=0)
        # We need to convert it to a list of list, but be able to go back
        nested_array = False

    # We need to create a new array with the number of features used for transform
    n_cols = 1
    n_rows, _ = Y.shape

    # Return array is the original features for
    Y_ret = np.empty((n_rows, n_cols), dtype=object)

    labels = Y[:, 0].astype("int64", copy=False)
    cats = np.array(fitted_values[encoder['input_features']])
    Y_ret[:, 0] = cats[labels]

    return Y_ret if nested_array else Y_ret[0]
