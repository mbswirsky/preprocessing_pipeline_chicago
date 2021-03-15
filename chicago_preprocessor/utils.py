import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .transformers import NullMaker


def column_separator(
    df, target_col, beat_cols, date_cols, null_thresh=0.1, cat_thresh=100
):
    """
    Identify lists of column types to be processed separately
    Columns are ID'd as null columns if proportion of nulls is above threshold
    Numeric and categorical columns are identified, excluding target
    and specified beat of occurence and date columns
    """

    df = NullMaker().fit_transform(df)

    null_cols = (
        (df.isnull().sum() / df.shape[0]).loc[lambda x: x > null_thresh].index.to_list()
    )

    num_cols = (
        df.drop(columns=null_cols, errors="ignore")
        .drop(columns=target_col, errors="ignore")
        .drop(columns=beat_cols, errors="ignore")
        .drop(columns=date_cols, errors="ignore")
        .select_dtypes(include="number")
        .columns.to_list()
    )

    cat_cols = (
        df.drop(columns=null_cols, errors="ignore")
        .drop(columns=target_col, errors="ignore")
        .drop(columns=beat_cols, errors="ignore")
        .drop(columns=date_cols, errors="ignore")
        .describe(include=["O"])
        .loc[
            :, lambda x: x.loc["unique"] < cat_thresh
        ]  # ignore columns with too many categories
        .columns.to_list()
    )

    return null_cols, num_cols, cat_cols


def sample_splitter(df, target_col, subsample_size=50000, test_size=0.2):
    """
    Subsets a dataframe, then splits it into X and y training and testing sets
    """

    df = df.sample(n=subsample_size)

    y = df[target_col]
    X = df.drop(columns=target_col, errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    return X_train, X_test, y_train, y_test
