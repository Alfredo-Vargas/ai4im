import pandas as pd
from imblearn.over_sampling import SMOTE

"""
The following preprocessing functions assume that you have no mixing of
datatypes for each pandas series on your dataframe.
"""


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    df[cols] = df[cols].apply(lambda x: (x - x.mean()) / x.std())
    return df


def drop_rows_with_only_nan_values(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    size = len(df)
    cols_to_drop = []
    for col in cols:
        if df[col].isna().sum() == size:
            cols_to_drop.append(col)
    return df.drop(cols_to_drop, axis=1)


def balance(df: pd.DataFrame, sr: pd.Series) -> tuple:
    smote = SMOTE(sampling_strategy="minority")
    resampled_features, resampled_labels = smote.fit_resample(df, sr)
    return (resampled_features, resampled_labels)


def check_imbalance(label: pd.Sereis) -> None:

        pv = np.count_nonzero(label) / len(label) * 100
    print(f"The imbalance profile has:\n {pv:.2f}% of valid products and \
        \n {100 - pv:.2f}% of invalid products")
