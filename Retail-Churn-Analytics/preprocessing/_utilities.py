

def _check_fitted(encoder):
    if not hasattr(encoder, "fit"):
        raise TypeError("{arg} is not an encoder instance.".format(arg=encoder))

    fitted = [
        v for v in vars(encoder) if v.endswith("_")
    ]
    if not fitted:
        # raise NotFittedError(msg % {"name": type(estimator).__name__})
        raise TypeError("This %(name)s instance is not fitted.".format(name=type(encoder).__name__))


def _columns_in_dataframe(columns, df):
    df_columns = set(df.columns)
    needed_cols = set([col.upper() for col in columns])

    required_cols_not_present = needed_cols - df_columns
    if len(required_cols_not_present):
        raise ValueError(
            f"Cannot find columns {required_cols_not_present} in the input dataframe. It must include the following "
            f"columns {needed_cols}. This is the columns found {df_columns}"
        )


def _generate_udf_encoder(encoder):
    _check_fitted(encoder)
    input_cols = encoder.input_cols

    udf_encoder = {"encoder": type(encoder).__name__, "nbr_features": len(input_cols), "input_features": input_cols,
                   "output_cols": encoder.output_cols, "fitted_values": {}}

    fitted_values = encoder.fitted_values_
    if not isinstance(input_cols, list):
        input_cols = [input_cols]

    if isinstance(fitted_values[input_cols[0]], dict):
        for k in fitted_values[input_cols[0]].keys():
            udf_encoder["fitted_values"][k] = []
            for col in input_cols:
                udf_encoder["fitted_values"][k].append(fitted_values[col][k])
    elif isinstance(fitted_values[input_cols[0]], list):
        udf_encoder["fitted_values"] = fitted_values

    if hasattr(encoder, 'handle_unknown'):
        udf_encoder["handle_unknown"] = encoder.handle_unknown
        if hasattr(encoder, 'unknown_value'):
            udf_encoder["unknown_value"] = encoder.unknown_value

    return udf_encoder
