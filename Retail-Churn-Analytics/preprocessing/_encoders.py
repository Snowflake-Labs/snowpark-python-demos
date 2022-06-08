from typing import Tuple, Union, List, Optional, Dict

from snowflake.snowpark import DataFrame
import snowflake.snowpark.functions as F
# from snowflake.snowpark import types as T
import json

from ._utilities import _check_fitted, _generate_udf_encoder, _columns_in_dataframe

__all__ = [
    "OneHotEncoder",
    "LabelEncoder",
    "OrdinalEncoder",
]


def _check_input_columns(df, input_columns):
    if not input_columns:
        input_columns = df.columns
    else:
        # Check if list
        if not isinstance(input_columns, list):
            input_columns = [input_columns]

    return input_columns


def _get_categories(df, categories, cat_cols):
    # {"COL1": ["cat1", "cat2", ...], "COL2": [...]}
    categories_ = []
    if categories == "auto":
        object_const = []
        for col in cat_cols:
            object_const.extend([F.lit(col), F.array_agg(F.to_varchar(col), is_distinct=True).within_group(
                F.to_varchar(F.col(col)).asc())])
        df_categories = df.select(F.object_construct(*object_const).as_("CATS"))
        categories_ = json.loads(df_categories.collect()[0][0])
    else:  # If categories has already been defined...
        # if self.handle_unknown == "error":
        # Check so we does not have new
        categories_ = categories

    return categories_


def _generate_label_where(encoder):

    col_exprs = []
    encode_cols = encoder.input_cols
    fitted_values = encoder.fitted_values_

    for col in encode_cols:
        with_expr = None
        for idx, cat in enumerate(fitted_values[col]):
            if type(with_expr) == F.CaseExpr:
                with_expr = with_expr.when(F.col(col) == F.lit(cat), F.lit(idx))
            else:
                with_expr = F.when(F.col(col) == F.lit(cat), F.lit(idx))
        if hasattr(encoder, 'handle_unknown'):
            if encoder.handle_unknown == "use_encoded_value":
                with_expr = with_expr.otherwise(F.lit(encoder.unknown_value))

        col_exprs.append(with_expr)

    return col_exprs


def _generate_inverse_sql(encoder):
    output_cols = encoder.output_cols

    fitted_values = encoder.fitted_values_

    input_cols = encoder.input_cols

    input_output = [list(i) for i in zip(input_cols, output_cols)]
    col_sql_expr = []
    for in_col, out_col in input_output:
        sql = f"AS_CHAR(PARSE_JSON('{json.dumps(fitted_values[in_col])}')[{out_col}])"
        col_sql_expr.append(F.sql_expr(sql))

    return col_sql_expr


class OneHotEncoder:
    def __init__(
            self,
            *,
            input_cols: Optional[Union[List[str], str]] = None,
            output_cols: Optional[Union[Dict, str]] = None,
            categories="auto",
            handle_unknown='ignore',
            drop_input_cols=True
    ):
        """
        Encode categorical features as a one-hot.

        When encoding using transform each category will be a column of their own.

        Unknown categories will always be ignored.

        :param input_cols: name of column or list of columns to encode
        :param output_cols: name of output column or list of output columns, need to be in the same order as the categories
        :param categories: "auto" or specified as a dict: {"COL1": [cat1, cat2, ...], "COL2": [cat1, cat2, ...], ...}
        :param handle_unknown:  Whether to 'ignored' or 'keep' when an unknown value is found during transform.
                                When set to ‘keep’, a new column is added during transform where unknown values get a 1
                                In inverse_transform, an unknown category will always be returned as NULL
        :param drop_input_cols: True/False if input columns should be dropped from the encoded DataFrame
        """
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.drop_input_cols = drop_input_cols

    def _check_output_columns(self):
        #  {"COL1": [cat1, cat2, ...], "COL2": [cat1, cat2, ...]}
        # output_columns ...
        cat_cols = {}
        needed_cols = 0
        output_cols = self.output_cols
        categories = self.fitted_values_
        input_cols = self.input_cols

        if output_cols:
            # Check so we have the same number of output columns as input columns
            if not len(output_cols) == len(input_cols):
                raise ValueError(f"Too few output columns provided. Have {len(output_cols)} need {len(input_cols)}")
            # Check so the total numer of columns per input column is equal to the number of categories
            tot_output_cols = sum(len(output_cols[feat]) for feat in output_cols)
            needed_cols = sum(len(output_cols[feat]) for feat in categories)
            if not needed_cols == tot_output_cols:
                raise ValueError(
                    f"Need the same number of output category columns as categories. Have {needed_cols} categories "
                    f"and {tot_output_cols} output category columns"
                )
            cat_cols = output_cols
        else:
            for col in input_cols:
                uniq_vals = categories[col]
                col_names = [col + '_' + val for val in uniq_vals]
                cat_cols[col] = col_names
                needed_cols += len(uniq_vals)

        # Snowflake can handle more columns,but it is depended on data type so let's keep it safe and limit to  3k
        if needed_cols > 3000:
            raise ValueError(
                "To many categories, maximum 3000 is allowed")

        return cat_cols

    def fit(self, df: DataFrame) -> object:
        """
        Fit the OneHotEncoder using df.

        :param df: Snowpark DataFrame used for getting the categories for each input column
        :return: Fitted encoder
        """
        #

        encode_cols = _check_input_columns(df, self.input_cols)
        self.input_cols = encode_cols

        self.fitted_values_ = _get_categories(df, self.categories, encode_cols)

        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Transform df using one-hot encoding, it will create one new column for each category found with fit.

        If drop_input_cols is True then the input columns are dropped from the returned DataFrame.

        :param df: Snowpark DataFrame to transform
        :return: Encoded Snowpark DataFrame
        """
        _check_fitted(self)

        encode_cols = self.input_cols

        output_cols = self._check_output_columns()
        self.output_cols = output_cols

        # Check for new categories?

        for col in encode_cols:
            uniq_vals = self.fitted_values_[col]
            col_names = output_cols[col]
            df = df.with_columns(col_names, [F.iff(F.col(col) == val, F.lit(1), F.lit(0)) for val in uniq_vals])
            if self.handle_unknown == 'keep':
                df = df.with_column(col + '__unknown', F.iff(~ F.col(col).in_(uniq_vals), F.lit(1), F.lit(0)))

            if self.drop_input_cols:
                df = df.drop(col)

        return df

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        Fit OneHotEncoder to df and transform the df, it will create one new column for each category found with fit.

        If drop_input_cols is True then the input columns are dropped from the returned DataFrame.
        :param df: Snowpark DataFrame to encode
        :return: Encoded Snowpark DataFrame
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        """
        Reverse the encoding.

        :param df: Snowpark DataFrame to reverse the encoding.
        :return: Reversed Snowpark DataFrame
        """
        _check_fitted(self)

        # We assume that columns in the input
        output_cols = self.output_cols
        # Verify that the df have the output columns
        verify_cols = []
        for k in output_cols:
            verify_cols.extend([col for col in output_cols[k]])

        _columns_in_dataframe(verify_cols, df)

        fitted_values = self.fitted_values_
        new_output_cols = []
        col_exprs = []
        for org_col in output_cols:
            new_output_cols.append(org_col)
            with_expr = None
            for idx, col in enumerate(output_cols[org_col]):
                if type(with_expr) == F.CaseExpr:
                    with_expr = with_expr.when(F.col(col) == F.lit(1), F.lit(fitted_values[org_col][idx]))
                else:
                    with_expr = F.when(F.col(col) == F.lit(1), F.lit(fitted_values[org_col][idx]))

            col_exprs.append(with_expr)

        ret_df = df.with_columns(new_output_cols, col_exprs)
        ret_df = ret_df.drop(verify_cols)
        if self.handle_unknown == 'keep':
            unk_cols = []
            for col in output_cols:
                unk_cols.append(col + '__unknown')
            ret_df = ret_df.drop(unk_cols)
        return ret_df

    def get_udf_encoder(self) -> Dict:
        """
        Returns the encoder as a dictionary object to be used with the udf_transform functions.

        :return: Dictionary containing fitted values
        """
        _check_fitted(self)

        return _generate_udf_encoder(self)


class OrdinalEncoder:
    def __init__(
            self,
            *,
            input_cols: Optional[Union[List[str], str]] = None,
            output_cols: Optional[Union[List[str], str]] = None,
            categories="auto",
            handle_unknown="ignored",
            unknown_value=None,
    ):
        """
        Encodes a string column of labels to a column of label indices. The indices are in [0, number of labels].

        By default, the labels are sorted alphabetically and numeric columns is cast to string.

        :param input_cols: name of column or list of columns to encode
        :param output_cols: name of output column or list of output columns, need to be in the same order as the categories
        :param categories: "auto" or specified as a dict {"COL1": [cat1, cat2, ...], "COL2": [cat1, cat2, ...], ...}
        :param handle_unknown: Whether to 'ignored' or use_encoded_value if a unknown category is present during transform.
                                When set to ‘use_encoded_value’, the encoded value of unknown categories will be set
                                to the value given for the parameter unknown_value.
                                In inverse_transform, an unknown category will always be returned as NULL
        :param unknown_value: When the parameter handle_unknown is set to ‘use_encoded_value’, this parameter is
                               required and will set the encoded value of unknown categories. It has to be distinct
                               from the values used to encode any of the categories

        """
        self.input_cols = input_cols
        self.output_cols = output_cols
        self.categories = categories
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value

    def fit(self, df: DataFrame) -> object:
        """
        Fit the OrdinalEncoder using df.

        :param df: Snowpark DataFrame used for getting the categories for each input column
        :return: Fitted encoder
        """
        encode_cols = _check_input_columns(df, self.input_cols)
        self.input_cols = encode_cols

        if self.handle_unknown == "use_encoded_value":
            if not isinstance(self.unknown_value, int):
                raise ValueError(f"unknown_value {self.unknown_value} is not a Integer")
        elif self.unknown_value is not None:
            raise ValueError(f"unknown_value can only be used with handle_unknown = 'use_encoded_value'")

        self.fitted_values_ = _get_categories(df, self.categories, encode_cols)

        if self.handle_unknown == "use_encoded_value":
            for cat in self.fitted_values_:
                if 0 <= self.unknown_value < len(cat):
                    raise ValueError(
                        f"The used value for unknown_value {self.unknown_value} is one of the values already used for "
                        f"encoding the categories."
                    )

        return self

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Transform input columns of df.

        :param df: Snowpark DataFrame to be transformed
        :return: A transformed Snowpark DataFrame
        """
        encode_cols = self.input_cols
        output_cols = self.output_cols

        # We will need as many output columns as input_cols
        if output_cols:
            if not isinstance(output_cols, list):
                output_cols = [output_cols]

            if len(output_cols) != len(encode_cols):
                raise ValueError(
                    f"Need the same number of output columns as categories  Have {len(encode_cols)} categories "
                    f"and {len(output_cols)} output columns"
                )
        else:
            output_cols = encode_cols

        self.output_cols = output_cols
        col_exprs = _generate_label_where(self)

        ret_df = df.with_columns(output_cols, col_exprs)

        return ret_df

    def fit_transform(self, df: DataFrame) -> DataFrame:
        """
        First fit then transoform using df
        :param df: Snowpark DataFrame used for fit and then transformed
        :return: A transformed Snowpark DataFrame
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, df: DataFrame) -> DataFrame:
        """
        Reverse the encoding.

        :param df: A Snowpark DataFrame with transformed columns
        :return: A reversed Snowpark DataFrame
        """
        _check_fitted(self)

        # We assume that columns in the input
        output_cols = self.output_cols
        # Verify that the df have the output columns
        _columns_in_dataframe(output_cols, df)

        col_sql_expr = _generate_inverse_sql(self)

        ret_df = df.with_columns(output_cols, col_sql_expr)
        return ret_df

    def get_udf_encoder(self) -> Dict:
        """
        Returns the encoder as a dictionary object to be used with the udf_transform functions.

        :return: Dictionary containing fitted values
        """
        _check_fitted(self)

        return _generate_udf_encoder(self)


class LabelEncoder:
    def __init__(self, input_col: str, output_col: str = None):
        """
        A label indexer that maps a string column of labels to a column of label indices.
        The indices are in [0, number of labels].

        :param input_col: Column
        :param output_col:
        """
        self.input_cols = input_col
        self.output_cols = output_col

    def fit(self, df: DataFrame):
        """

        :param df: DataFrame
        :return:
        """
        # y = column_or_1d(y, warn=True)
        # self.classes_ = _unique(y)
        # check that y is existing in the df
        #
        input_col = self.input_cols
        self.fitted_values_ = _get_categories(df, "auto",[input_col])

        return self.fitted_values_

    def transform(self, df: DataFrame):

        _check_fitted(self)

        input_col = self.input_cols
        output_col = self.output_cols

        if not output_col:
            output_col = input_col
        #
        self.input_cols = [input_col]
        self.output_cols = [output_col]

        col_exprs = _generate_label_where(self)
        encoded_df = df.with_columns([output_col], col_exprs)
        return encoded_df

    def fit_transform(self, df: DataFrame):
        """

        :param df:
        :return:
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, df: DataFrame):
        encoded_df = df

        _check_fitted(self)

        # We assume that columns in the input
        output_cols = self.output_cols
        # Verify that the df have the output columns
        _columns_in_dataframe(output_cols, df)

        col_sql_expr = _generate_inverse_sql(self)

        ret_df = df.with_columns(output_cols, col_sql_expr)

        return ret_df

    def get_udf_encoder(self) -> Dict:
        """
        Returns the encoder as a dictionary object to be used with the udf_transform functions.

        :return: Dictionary containing fitted values
        """
        _check_fitted(self)

        return _generate_udf_encoder(self)
