from typing import Tuple, Union, List, Optional, Dict

from snowflake.snowpark import DataFrame
import snowflake.snowpark.functions as F
from snowflake.snowpark import types as T

import json
from scipy import stats

from ._utilities import _check_fitted, _generate_udf_encoder, _columns_in_dataframe

__all__ = [
    "MinMaxScaler",
    "StandardScaler",
    "MaxAbsScaler",
    "RobustScaler",
    "Normalizer",
    "Binarizer",
]

def _get_numeric_columns(df: DataFrame) -> List:
    numeric_types = [T.DecimalType, T.LongType, T.DoubleType, T.FloatType, T.IntegerType]
    numeric_cols = [c.name for c in df.schema.fields if type(c.datatype) in numeric_types]

    if len(numeric_cols) == 0:
        raise ValueError(
            "No numeric columns in the provided DataFrame"
        )
    return numeric_cols


def _fix_scale_columns(df, scale_columns) -> List:
    if not scale_columns:
        # Get all numeric columns
        scale_columns = _get_numeric_columns(df)
    else:
        # Check if list
        if not isinstance(scale_columns, list):
            scale_columns = [scale_columns]

    return scale_columns


def _check_output_columns(output_cols, input_columns) -> List:
    if output_cols:
        if not isinstance(output_cols, list):
            output_cols = [output_cols]
        # Check so we have the same number of output columns as input columns
        if not len(input_columns) == len(output_cols):
            raise ValueError(
                f"Need the same number of output columns as input columns  Got {len(input_columns)} "
                f"input columns and {len(input_columns)} output columns"
            )
    else:
        output_cols = input_columns

    return output_cols


class MinMaxScaler:
    def __init__(self, *, feature_range: Optional[Tuple[int, int]] = (0, 1),
                 input_cols: Optional[Union[List[str], str]] = None,
                 output_cols: Optional[Union[List[str], str]] = None):
        """
        Scale features to a fixed range, e.g. between zero and one, using Min and Max.

        :param feature_range: Range of transformed data. Defaults to (0, 1).
        :param input_cols: Column or columns to scale, if not provided all numeric columns in the dataframe will be used
        :param output_cols: Names of scaled columns, if not provided output columns will have the same names as the input columns
        """

        self.feature_range = feature_range
        self.input_cols = input_cols
        self.output_cols = output_cols

    def fit(self, df: DataFrame):
        """
        Calculates min and max on input columns for df for later scaling.

        :param df: Snowpark DataFrame to be scaled
        :return: fitted scaler
        """
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(feature_range)
            )

        # If not provided columns to scale
        scale_columns = _fix_scale_columns(df, self.input_cols)
        self.input_cols = scale_columns

        if len(scale_columns) == 0:
            raise ValueError(
                "No columns to fit, the DataFrame has no numeric columns")

        # Maybe better to return {"COL1": {"MIN": nbr, "MAX": nbr }}
        select_calc = []
        range_cols_exp = []
        range_cols_names = []
        scale_cols_exp = []
        scale_cols_names = []
        min_cols_exp = []
        min_cols_names = []
        obj_const_log = []
        for col in scale_columns:
            select_calc.extend([F.max(F.col(col)).as_("max_" + col), F.min(F.col(col)).as_("min_" + col)])
            range_cols_names.extend(["range_" + col])
            range_cols_exp.extend([(F.col("max_" + col) - F.col("min_" + col))])
            scale_cols_names.extend(["scale_" + col])
            scale_cols_exp.extend([(F.lit(feature_range[1]) - F.lit(feature_range[0])) / F.col("range_" + col)])
            min_cols_names.extend(["min__" + col])
            min_cols_exp.extend([(F.lit(feature_range[0]) - F.col("min_" + col) * F.col("scale_" + col))])
            obj_const_log.extend([F.lit(col), F.object_construct(F.lit("max"), F.col("max_" + col),
                                                                 F.lit("min"), F.col("min_" + col),
                                                                 F.lit("range"), F.col("range_" + col),
                                                                 F.lit("scale"), F.col("scale_" + col),
                                                                 F.lit("min_"), F.col("min__" + col))])

        df_fitted_values = df.select(select_calc) \
            .with_columns(range_cols_names, range_cols_exp) \
            .with_columns(scale_cols_names, scale_cols_exp) \
            .with_columns(min_cols_names, min_cols_exp) \
            .select(F.object_construct(*obj_const_log))

        fitted_values = json.loads(df_fitted_values.collect()[0][0])
        # Store the column order?
        self.fitted_values_ = fitted_values

        return self

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Calculates min and max on input columns for df and then use them for scaling.

        :param df: Snowpark DataFrame
        :return: Snowpark DataFrame with scaled columns
        """
        return self.fit(df).transform(df)

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Scales input columns and adds the scaled values in ooutput columns.

        :param df: Snowpark DataFrame to be scaled.
        :return: Snowpark DataFrame with scaled columns

        """
        # Check if fitted otherwise raise error!
        _check_fitted(self)

        # If not provided columns to scale
        scale_columns = self.input_cols
        if not isinstance(scale_columns, list):
            scale_columns = [scale_columns]

        output_cols = _check_output_columns(self.output_cols, scale_columns)
        self.output_cols = output_cols

        fitted_values = self.fitted_values_
        # Do the scaling
        trans_df = df.with_columns(output_cols,
                                   [((F.col(col) * F.lit(fitted_values[col]["scale"])) + fitted_values[col]["min_"])
                                    for col in scale_columns])

        return trans_df

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        """
        Undo scaling of output columns in provided DataFrame.

        :param df: Snowpark DataFrame with scaled output columns
        :return: Snowpark DataFrame with undone scaling
        """
        # Check if fitted otherwise raise error!
        _check_fitted(self)

        input_cols = self.input_cols
        output_cols = self.output_cols
        # Check so that the output columns exists in the provided DataFrame
        _columns_in_dataframe(output_cols, df)
        input_output = [list(i) for i in zip(input_cols, output_cols)]
        #
        # Instead of creating a specific out_fitted_values We can assume that the output_cols are in the same order
        # as the input
        fitted_values = self.fitted_values_

        trans_df = df.with_columns(output_cols,
                                   [((F.col(in_out[1]) - fitted_values[in_out[0]]["min_"]) / F.lit(
                                       fitted_values[in_out[0]]["scale"]))
                                    for in_out in input_output])

        return trans_df

    def get_udf_encoder(self) -> Dict:
        """
        Returns the encoder as a dictionary object to be used with the udf_transform functions.

        :return: Dictionary containing fitted values
        """
        _check_fitted(self)

        return _generate_udf_encoder(self)


class StandardScaler:
    def __init__(self, *, with_mean=True, with_std=True,
                 input_cols: Optional[Union[List[str], str]] = None,
                 output_cols: Optional[Union[List[str], str]] = None):
        """
        Standardize features by removing the mean and scaling to unit variance.

        :param with_mean: If True, center the data before scaling.
        :param with_std: If True, scale the data to unit standard deviation.
        :param input_cols: Column or columns to scale, if not provided all numeric columns in the dataframe will be used
        :param output_cols: Names of scaled columns, if not provided output columns will have the same names as the input columns

        """
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.with_mean = with_mean
        self.with_std = with_std

    def fit(self, df):
        """
        Compute the mean and std to be used for later scaling.

        :param df: Snowpark DataFrame to be scaled
        :return: fitted encoder
        """
        # Not using sample_weight=None for now!

        # Validate data
        scale_columns = _fix_scale_columns(df, self.input_cols)
        self.input_cols = scale_columns

        if len(scale_columns) == 0:
            raise ValueError(
                "No columns to fit, the DataFrame has no numeric columns")

        # if sample_weight is not None:
        # sample_weight is one per row

        if not self.with_mean and not self.with_std:
            fitted_values = {}
            for col in scale_columns:
                fitted_values[col] = {}
                fitted_values[col]['mean'] = 0
                fitted_values[col]['scale'] = 1
        else:
            obj_const_log = []
            for col in scale_columns:
                obj_const_log.extend([F.lit(col), F.object_construct(F.lit("mean"), F.mean(F.col(col)),
                                                                     F.lit("scale"), F.stddev(F.col(col)))])

            df_fitted_values = df.select(F.object_construct(*obj_const_log))

            fitted_values = json.loads(df_fitted_values.collect()[0][0])

            if not self.with_std:
                #    self.scale_ = np.array(mean_var["stddev"])
                for col in scale_columns:
                    fitted_values[col]["scale"] = 1
            if not self.with_mean:
                for col in scale_columns:
                    fitted_values[col]["mean"] = 0

        self.fitted_values_ = fitted_values

        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Scales input columns and adds the scaled values in output columns.

        :param df: Snowpark DataFrame to be scaled.
        :return: Snowpark DataFrame with scaled columns
        """
        # Need check if fitted
        _check_fitted(self)

        scale_columns = self.input_cols

        output_cols = _check_output_columns(self.output_cols, scale_columns)
        self.output_cols = output_cols
        fitted_values = self.fitted_values_

        trans_df = df.with_columns(output_cols,
                                   [((F.col(col) - F.lit(fitted_values[col]["mean"])) / fitted_values[col]["scale"])
                                    for col in scale_columns])

        return trans_df

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Compute the mean and std and scales the DataFrame using those values.

        :param df: Snowpark DataFrame to be scaled.
        :return: Snowpark DataFrame with scaled columns

        """
        return self.fit(df).transform(df)

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        """
        Undo scaling of output columns in provided DataFrame.

        :param df: Snowpark DataFrame with scaled output columns
        :return: Snowpark DataFrame with undone scaling
        """
        # Check if fitted otherwise raise error!
        _check_fitted(self)
        input_cols = self.input_cols
        output_cols = self.output_cols

        # Check so that the output columns exists in the provided DataFrame
        _columns_in_dataframe(output_cols, df)
        input_output = [list(i) for i in zip(input_cols, output_cols)]

        fitted_values = self.fitted_values_

        trans_df = df.with_columns(output_cols,
                                   [((F.col(col[1]) * fitted_values[col[0]]["scale"]) + F.lit(
                                       fitted_values[col[0]]["mean"]))
                                    for col in input_output])

        return trans_df

    def get_udf_encoder(self):
        """
        Returns the encoder as a dictionary object to be used with the udf_transform functions.

        :return: Dictionary containing fitted values
        """
        _check_fitted(self)

        return _generate_udf_encoder(self)


class MaxAbsScaler:
    def __init__(self, *,
                 input_cols: Optional[Union[List[str], str]] = None,
                 output_cols: Optional[Union[List[str], str]] = None):
        """
        Scale each feature by its maximum absolute value.

        :param input_cols: Column or columns to scale, if not provided all numeric columns in the dataframe will be used
        :param output_cols: Names of scaled columns, if not provided output columns will have the same names as the input columns

        """
        self.input_cols = input_cols
        self.output_cols = output_cols

    def fit(self, df: DataFrame):
        """
        Gets the maximum absolute value for each input column to be used for later scaling.

        :param df: Snowpark DataFrame to be scaled
        :return: fitted scaler
        """
        scale_columns = _fix_scale_columns(df, self.input_cols)
        self.input_cols = scale_columns

        obj_const_log = []
        for col in scale_columns:
            obj_const_log.extend([F.lit(col), F.object_construct(F.lit("max_abs"), F.abs(F.max(F.col(col))),
                                                                 F.lit("scale"), F.iff(
                    F.abs(F.max(F.col(col))) > (F.lit(10) * F.pow(2, -52)), F.abs(F.max(F.col(col))), F.lit(1)))])

        df_fitted_values = df.select(F.object_construct(*obj_const_log))

        fitted_values = json.loads(df_fitted_values.collect()[0][0])
        self.fitted_values_ = fitted_values

        return self

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """

        :param df: DataFrame to transform.
        :return:
        """
        return self.fit(df).transform(df)

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Scales input columns and adds the scaled values in output columns.

        :param df: Snowpark DataFrame to be scaled.
        :return: Snowpark DataFrame with scaled columns
        """
        _check_fitted(self)

        scale_columns = self.input_cols

        output_cols = _check_output_columns(self.output_cols, scale_columns)
        self.output_cols = output_cols

        fitted_values = self.fitted_values_

        trans_df = df.with_columns(output_cols,
                                   [(F.col(col) / fitted_values[col]["scale"]) for col in scale_columns])
        return trans_df

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        """
        Scale back the data to the orginial values.
        The inversed columns will be the columns set by output_columns.

        :param df: DataFrame to inverse
        :return: DataFrame with inversed columns
        """
        _check_fitted(self)
        output_cols = self.output_cols
        input_cols = self.input_cols
        # Check so that the output columns exists in the provided DataFrame
        _columns_in_dataframe(output_cols, df)
        input_output = [list(i) for i in zip(input_cols, output_cols)]

        fitted_values = self.fitted_values_

        trans_df = df.with_columns(output_cols,
                                   [(F.col(col[1]) * fitted_values[col[0]]["scale"]) for col in input_output])
        return trans_df

    def get_udf_encoder(self):
        """
        Returns the encoder as a dictionary object to be used with the udf_transform functions.

        :return: Dictionary containing fitted values
        """
        _check_fitted(self)

        return _generate_udf_encoder(self)


class RobustScaler:
    def __init__(
            self,
            *,
            with_centering=True,
            with_scaling=True,
            quantile_range=(25.0, 75.0),
            unit_variance=False,
            input_cols: Optional[Union[List[str], str]] = None,
            output_cols: Optional[Union[List[str], str]] = None,
    ):
        """
        Scale features using statistics that are robust to outliers.

        This scaler scales by remove the median and scales the data according to the quantile range
        (defaults to IQR: Interquartile Range) The IQR is the range between the 1st quartile (25th quantile)
        and the 3rd quartile (75th quantile).

        :param with_centering: If True, center the data before scaling
        :param with_scaling: If True, scale the data to interquartile range.
        :param quantile_range: Quantile range used to calculate scale_. By default this is equal to the IQR
        :param unit_variance:  If True, scale data so that normally distributed features have a variance of 1
        :param input_cols: Column or columns to scale, if not provided all numeric columns in the dataframe will be used
        :param output_cols: Names of scaled columns, if not provided output columns will have the same names as the input columns

        """
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.unit_variance = unit_variance

    def fit(self, df: DataFrame):
        """
        Compute the median and quantiles to be used for scaling.
        :param df: Snowpark DataFrame to be scaled
        :return: fitted scaler
        """
        # Validate data
        scale_columns = _fix_scale_columns(df, self.input_cols)
        self.input_cols = scale_columns

        q_min, q_max = self.quantile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError("Invalid quantile range: %s" % str(self.quantile_range))

        # Convert it into percentiles values supported by Snowflake
        snf_q_range = [q / 100 for q in list(self.quantile_range)]
        obj_const_log = []
        for col in scale_columns:
            key_vals = []

            if self.with_centering:
                key_vals.extend([F.lit("center"), F.median(F.col(col))])
            else:
                key_vals.extend([F.lit("center"), F.lit(0)])

            if self.with_scaling:
                scaled_sql = f"(PERCENTILE_CONT({snf_q_range[1]}) WITHIN GROUP (ORDER BY {col}) - " \
                             f"PERCENTILE_CONT({snf_q_range[0]}) WITHIN GROUP (ORDER BY {col}))"

                if self.unit_variance:
                    adjust = stats.norm.ppf(q_max / 100.0) - stats.norm.ppf(q_min / 100.0)
                    scaled_sql = scaled_sql + f"/{adjust}"

                key_vals.extend([F.lit("scale"), F.sql_expr(scaled_sql)])
            else:
                key_vals.extend([F.lit("scale"), F.lit(1)])
            obj_const_log.extend([F.lit(col), F.object_construct(*key_vals)])

        df_fitted_values = df.select(F.object_construct(*obj_const_log))
        fitted_values = json.loads(df_fitted_values.collect()[0][0])
        self.fitted_values_ = fitted_values

        return self

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Compute the median and quantiles and then scales the DataFrame with those.

        :param df: Snowpark DataFrame to be scaled.
        :return: Snowpark DataFrame with scaled columns

        """
        return self.fit(df).transform(df)

    def transform(self, df: DataFrame):
        """
        Scales input columns and adds the scaled values in ooutput columns.

        :param df: Snowpark DataFrame to be scaled.
        :return: Snowpark DataFrame with scaled columns

        """
        _check_fitted(self)

        scale_columns = self.input_cols

        output_cols = _check_output_columns(self.output_cols, scale_columns)
        self.output_cols = output_cols
        fitted_values = self.fitted_values_

        trans_df = df.with_columns(output_cols,
                                   [((F.col(col) - fitted_values[col]["center"]) / fitted_values[col]["scale"])
                                    for col in scale_columns])

        return trans_df

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        """
        Scale back the data to the orginial values.
        The inversed columns will be the columns set by output_columns.

        :param df: DataFrame to inverse
        :return: DataFrame with inversed columns
        """

        _check_fitted(self)

        output_cols = self.output_cols
        _columns_in_dataframe(output_cols, df)

        input_cols = self.input_cols
        input_output = [list(i) for i in zip(input_cols, output_cols)]

        fitted_values = self.fitted_values_

        trans_df = df.with_columns(output_cols,
                                   [((F.col(col[1]) * fitted_values[col[0]]["scale"]) + fitted_values[col[0]]["center"])
                                    for col in input_output])
        return trans_df

    def get_udf_encoder(self):
        """
        Returns the encoder as a dictionary object to be used with the udf_transform functions.

        :return: Dictionary containing fitted values
        """
        _check_fitted(self)

        return _generate_udf_encoder(self)


class Normalizer:

    def _get_sql_norms(self, norm):

        if norm not in ("l1", "l2", "max"):
            raise ValueError("'%s' is not a supported norm" % norm)

        scale_columns = self.input_cols

        if norm == "l1":
            sql_abs = []
            for col in scale_columns:
                sql_col_abs = 'ABS({0})'.format(col)
                sql_abs.append(sql_col_abs)

            sql_sum = ''
            for col in sql_abs:
                if sql_sum != '':
                    sql_sum = sql_sum + '+'
                sql_sum = sql_sum + col

            sql_norms = '({0})'.format(sql_sum)
        elif norm == "l2":
            sql_square = []
            for col in scale_columns:
                sql_col_sqr = 'SQUARE({0})'.format(col)
                sql_square.append(sql_col_sqr)

            sql_sum = ''
            for col in sql_square:
                if sql_sum != '':
                    sql_sum = sql_sum + '+'
                sql_sum = sql_sum + col

            sql_norms = 'SQRT({0})'.format(sql_sum)

        elif norm == "max":
            sql_abs = []
            for col in scale_columns:
                sql_col_abs = 'ABS({0})'.format(col)
                sql_abs.append(sql_col_abs)

            sql_sum = ''
            for col in sql_abs:
                if sql_sum != '':
                    sql_sum = sql_sum + ','
                sql_sum = sql_sum + col

            sql_norms = 'GREATEST({0})'.format(sql_sum)
        return sql_norms

    def __init__(self,
                 *,
                 norm="l2",
                 input_cols: Optional[Union[List[str], str]] = None,
                 output_cols: Optional[Union[List[str], str]] = None, ):
        """
        Normalize individually to unit norm.

        :param norm: The norm to use to normalize each non zero data. If norm=’max’ is used, values will be rescaled
        by the maximum of the absolute values.
        :param input_cols: Column or columns to scale, if not provided all numeric columns in the dataframe will be used
        :param output_cols: Names of scaled columns, if not provided output columns will have the same names as the input columns

        """
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.norm = norm

    def fit(self, df: DataFrame):
        """
        Do nothing. Only verifies the input columns.

        :param df: Snowpark DataFrame to be scaled
        :return: fitted scaler
        """
        scale_columns = _fix_scale_columns(df, self.input_cols)
        self.input_cols = scale_columns

        sql_norms = self._get_sql_norms(self.norm)

        #df_fitted_values = df.with_columns(output_cols, [F.col(col) / F.sql_expr(sql_norms) for col in scale_columns])
        #df_fitted_values = df.select(F.object_construct(*obj_const_log))
        #fitted_values = json.loads(df_fitted_values.collect()[0][0])
        #fitted_values =
        self.fitted_values_ = {'norms_sql': sql_norms}

        self.fitted_ = True

        return self

    def fit_transform(self, df: DataFrame):
        """

        :param df: Snowpark DataFrame to be scaled
        :return: Scaled Snowpark DataFrame
        """
        return self.fit(df).transform(df)

    def transform(self, df: DataFrame):
        """
        Scale each input column of df to unit norm.

        :param df: Snowpark DataFrame to be scaled
        :return: Scaled Snowpark DataFrame
        """
        _check_fitted(self)

        scale_columns = self.input_cols
        output_cols = _check_output_columns(self.output_cols, scale_columns)
        self.output_cols = output_cols
        sql_norms = self.fitted_values_["norms_sql"]

        df_ret = df.with_columns(output_cols, [F.col(col) / F.sql_expr(sql_norms) for col in scale_columns])

        return df_ret

    def get_udf_encoder(self):
        """
        Returns the encoder as a dictionary object to be used with the udf_transform functions.

        :return: Dictionary containing fitted values
        """
        _check_fitted(self)
        input_cols = self.input_cols

        udf_encoder = {"encoder": type(self).__name__, "nbr_features": len(input_cols), "input_features": input_cols,
                       "output_cols": self.output_cols, "fitted_values": {"norm": self.norm}}

        return udf_encoder


class Binarizer:
    def __init__(self, *, threshold=0.0,
                 input_cols: Optional[Union[List[str], str]] = None,
                 output_cols: Optional[Union[List[str], str]] = None, ):
        """
        Binarize data (set feature values to 0 or 1) according to a threshold.

        :param threshold: Feature values below or equal to this are replaced by 0, above it by 1.
        :param input_cols: Column or columns to scale, if not provided all numeric columns in the dataframe will be used
        :param output_cols: Names of scaled columns, if not provided output columns will have the same names as the input columns
        """
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.threshold = threshold

    def fit(self, df: DataFrame):
        """
        Do nothing. Only verifies the input columns.

        :param df: Snowpark DataFrame to be scaled
        :return: fitted scaler
        """
        scale_columns = _fix_scale_columns(df, self.input_cols)
        self.input_cols = scale_columns
        self.fitted_ = True

        return self

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Binarize input columns of df (set feature values to 0 or 1) according to threshold.

        :param df: Snowpark DataFrame to be scaled
        :return: Snowpark DataFrame with binarized output columns
        """
        return self.fit(df).transform(df)

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Binarize input columns of df (set feature values to 0 or 1) according to threshold.

        :param df: Snowpark DataFrame to be scaled
        :return: Snowpark DataFrame with binarized output columns
        """
        _check_fitted(self)
        scale_columns = self.input_cols
        output_cols = _check_output_columns(self.output_cols, scale_columns)
        self.output_cols = output_cols

        # All values that are larger than threshold should be 1 and others 0
        df_ret = df.with_columns(output_cols, [F.iff(F.col(col) > F.lit(self.threshold), F.lit(1), F.lit(0))
                                               for col in scale_columns])
        return df_ret

    def get_udf_encoder(self):
        """
        Returns the encoder as a dictionary object to be used with the udf_transform functions.

        :return: Dictionary containing fitted values
        """
        _check_fitted(self)

        input_cols = self.input_cols

        udf_encoder = {"encoder": type(self).__name__, "nbr_features": len(input_cols), "input_features": input_cols,
                       "output_cols": self.output_cols, "fitted_values": {"threshold": self.threshold}}

        return udf_encoder
