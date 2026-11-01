# Custom scikit-learn transformers for the Suwayhib preprocessing pipeline.
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Learner recording length divided by reference recording length.
def add_engineered_features(df):
    # copy so the caller's table is not changed
    df = df.copy()
    learner_length = df["new_duration_ms"]
    reference_length = df["orig_duration_ms"]
    df["rec_len_ratio"] = learner_length / reference_length
    return df


# Adds rec_len_ratio. Learns nothing in fit.
class FeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # only remember the input column names; nothing is estimated
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        return add_engineered_features(X)

    # needed so set_output(transform="pandas") knows the output column names
    def get_feature_names_out(self, input_features=None):
        return np.append(self.feature_names_in_, "rec_len_ratio")


# Features of the previous and next word in the same recording (lag and lead).
# Learns nothing in fit. The first and last word of a recording have no previous or
# next word, so they reuse their own value; that keeps "this is the first word" from
# leaking back in through a missing-value marker (word_idx was dropped for that reason).
class NeighborFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, columns=("dtw_cost", "dur_ratio")):
        self.columns = columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def _new_names(self):
        names = []
        for column in self.columns:
            names.append(f"{column}_prev")
            names.append(f"{column}_next")
        return names

    def transform(self, X):
        X = X.copy()
        # shift inside each recording, in word order; the original row order is kept
        ordered = X.sort_values(["recording_id", "word_idx"])
        by_recording = ordered.groupby("recording_id")
        for column in self.columns:
            previous = by_recording[column].shift(1)
            following = by_recording[column].shift(-1)
            X[f"{column}_prev"] = previous.reindex(X.index).fillna(X[column])
            X[f"{column}_next"] = following.reindex(X.index).fillna(X[column])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.append(self.feature_names_in_, self._new_names())


# Drops a fixed list of columns decided in Phase 1 (not learned from data).
class ColumnDropper(BaseEstimator, TransformerMixin):

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        # skip names that are not in X, so the same list works on any table
        columns_present = []
        for column in self.columns:
            if column in X.columns:
                columns_present.append(column)
        return X.drop(columns=columns_present)

    # the input columns minus the dropped ones, in their original order
    def get_feature_names_out(self, input_features=None):
        kept = []
        for column in self.feature_names_in_:
            if column not in self.columns:
                kept.append(column)
        return np.asarray(kept)


# Applies log1p to the right-skewed columns. Learns nothing in fit.
class Log1pTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, columns=()):
        self.columns = columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        X = X.copy()
        # columns not in self.columns pass through unchanged
        for column in self.columns:
            # log1p is undefined below -1; these features should never be negative
            non_negative = X[column].clip(lower=0)
            X[column] = np.log1p(non_negative)
        return X

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_


# Caps each column at [Q1 - k*IQR, Q3 + k*IQR]. The fences come from the training data only.
class IQRWinsorizer(BaseEstimator, TransformerMixin):

    # k=1.5 is Tukey's usual fence; k=3.0 is the "far-out" fence, which clips only extreme values
    def __init__(self, k=3.0):
        self.k = k

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        # quantile works column by column, so q1 and q3 hold one value per feature
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        # one lower and one upper fence per column, stored for transform
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X):
        # rebuild the DataFrame in case a step before passed a plain array
        X = pd.DataFrame(X, columns=self.feature_names_in_)
        # axis=1 lines each fence up with its own column
        X_capped = X.clip(lower=self.lower_, upper=self.upper_, axis=1)
        return X_capped

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_in_
