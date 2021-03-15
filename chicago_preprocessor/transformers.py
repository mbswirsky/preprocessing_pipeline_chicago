import pandas as pd
from sklearn.base import TransformerMixin

class NullMaker(TransformerMixin):
    """
    Finds values that should be nulls and replaces with null
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xnulls = X.replace(['UNABLE TO DETERMINE', 'NOT APPLICABLE', 'UNKNOWN'], np.nan)
        return Xnulls


class BeatFormatter(TransformerMixin):
    """
    Roll up police beat of occurence value to two digits
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        dfs=[]
        X = pd.DataFrame(X)
        for col in X:
            Xbeat = X[col].astype(str).str[:-2].str.zfill(4).str[:2]
        dfs.append(Xbeat)
        return pd.concat(dfs, axis=1)


class DateFormatter(TransformerMixin):
    """
    Convert string represented dates into pandas timestamps
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xdate = X.apply(pd.to_datetime)
        return Xdate


class DateEncoder(TransformerMixin):
    """
    Extract year, month, weekday, day of month, and hour of day from timestamps
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # assumes X is a DataFrame of datetime dtypes
        dfs=[]
        for col in X:
            dt = X[col].dt
            df_dt = pd.concat([dt.year, dt.month, dt.dayofweek, dt.day, dt.hour], axis=1)
            dfs.append(df_dt)
        return pd.concat(dfs, axis=1)
