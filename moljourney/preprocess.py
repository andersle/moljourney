"""Methods for preprocessing (nan, variance, correlations)."""
import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer

LOGGER = logging.getLogger(__name__)


def count_nan(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Count the number of NaNs in the data frame."""
    nan_count = dataframe.isnull().sum()
    nan_count.name = "nan-count"
    nan_fract = dataframe.isnull().mean()
    nan_fract.name = "nan-fraction"
    nan_data = pd.concat([nan_count, nan_fract], axis=1)
    nan_data.index.name = "feature"
    nan_data = nan_data.reset_index()
    return nan_data


def run_impute(
    dataframe: pd.DataFrame,
    imputer_str: str = "",
    imputer: type[SimpleImputer] | None = None,
) -> pd.DataFrame:
    """Run imputation on numeric part of data frame.

    Parameters
    ----------
    dataframe : object like pd.DataFrame
        The data to impute. Note: We only select the
        numeric part of the data here.
    imputer_str : string, optional
        String used to choose a predefined imputer - a quick way
        to test imputation. You may want to define your own imputer.
    imputer : object like SimpleImputer, optional
        The imputer we will use here. Assumed to
        have a fit_transform method.
    """
    if imputer is None:
        imputers = {
            "impute-mean": SimpleImputer(
                missing_values=np.nan, strategy="mean"
            ),
            "impute-median": SimpleImputer(
                missing_values=np.nan, strategy="median"
            ),
            "impute-knn": KNNImputer(
                missing_values=np.nan, n_neighbors=5, weights="distance"
            ),
        }
        imputer = imputers.get(imputer_str, None)
        if imputer is None:  # Unknown selection for impute-XXX
            raise ValueError(
                f'Unknown imputer "{imputer_str}", expected one of: '
                "{list(imputers.keys())}"
            )
    numeric = dataframe.select_dtypes(include="number")
    non_numeric = dataframe.select_dtypes(exclude="number")
    numeric_frame = pd.DataFrame(
        imputer.fit_transform(numeric),
        columns=numeric.columns,
        index=numeric.index,
    )
    return pd.concat([non_numeric, numeric_frame], axis=1)


def remove_all_nan_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Remove all columns with NaNs."""
    nan_count = data.isnull().sum()
    columns_nan = set(nan_count[nan_count > 0].index)
    if len(columns_nan) > 0:
        LOGGER.info("Removed %i column(s) with NaNs.", len(columns_nan))
        LOGGER.debug("Columns removed: %s", sorted(columns_nan))
    return data.dropna(axis=1)


def remove_all_nan_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Remove all rows with NaNs."""
    drop = data.dropna(axis=0)

    diff = data.index.difference(drop.index)

    if len(diff) > 0:
        LOGGER.info("Removed %i sample(s) with NaNs.", len(diff))
        removed_idx = sorted(list(diff))
        LOGGER.debug("Columns removed: %s", removed_idx)
    return drop


def remove_nan_threshold(
    data: pd.DataFrame, threshold: float = 0.2
) -> pd.DataFrame:
    """Remove columns where the fraction of NaN > threshold."""
    nan_fraction = data.isnull().mean()
    remove = set(nan_fraction[nan_fraction > threshold].index)
    # Remove above the given threshold:
    if len(remove) > 0:
        LOGGER.info(
            "Removed %i columns(s) with NaN fraction > %f.",
            len(remove),
            threshold,
        )
        LOGGER.debug("Column(s) removed: %s", sorted(remove))
    return data.drop(columns=list(remove))


def preprocess_nan(
    data: pd.DataFrame,
    handle_nan: str | type[SimpleImputer] = "remove",
    remove_threshold: float = 0.20,
    force_numeric: bool = False,
    include_inf: bool = True,
) -> pd.DataFrame:
    """Remove or impute NaNs.

    If handle_nan is "remove", columns with NaNs are removed. Imputation
    can be carried out by setting this string to "impute-mean",
    "impute-median" or "impute-knn". This is probably not what you want,
    but it is included as a "quick" way to see the effect of imputation.
    Passing an object will use that object for imputation if
    it has a method "fit_transform". The remove_threshold parameter
    is used to remove columns based on the fraction of NaNs in the column.

    Parameters
    ----------
    data : object like pd.DataFrame
        The data we will process
    handle_nan : string or object like SimpleImputer.
        This is how we handle NaNs for columns where the fraction
        is missing numbers is smaller than remove_threshold.
        If a string is
        given this is interpreted as one of the predefined
        methods given in run_impute.
    remove_threshold : float, optional
        Columns with a fraction of missing numbers larger than this
        threshold will be removed.
    force_numeric : bool, optional
        If True, this method will transform all columns to numeric
        via pd.to_numeric.
    include_inf : bool, optional
        Interpret np.inf and -np.inf as a NaN if True.

    Note
    ----
    This method will make a copy of the input dataframe.
    """
    dataframe = data.copy()
    if force_numeric:
        dataframe = dataframe.apply(pd.to_numeric, errors="coerce", axis=1)
    if include_inf:
        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)

    # If handle_nan is "remove": remove irrespective of
    # the threshold.
    if isinstance(handle_nan, str) and handle_nan == "remove":
        return remove_all_nan_columns(dataframe)
    if isinstance(handle_nan, str) and handle_nan == "remove-samples":
        return remove_all_nan_rows(dataframe)
    # Remove based on the fraction of NaNs:
    dataframe = remove_nan_threshold(dataframe, threshold=remove_threshold)

    # Check if we still have some NaNs to impute:
    nan_count = dataframe.isnull().sum()
    columns_nan = list(set(nan_count[nan_count > 0].index))

    if len(columns_nan) == 0:
        return dataframe

    column_list = f"{sorted(columns_nan)}"
    column_txt = f"{len(columns_nan)} column(s)"

    if isinstance(handle_nan, str):
        if handle_nan.startswith("impute"):
            LOGGER.info(
                'Imputed %s with predefined "%s" method.',
                column_txt,
                handle_nan,
            )
            LOGGER.debug("Column(s) imputed: %s", column_list)
            return run_impute(dataframe, imputer_str=handle_nan)
        else:
            LOGGER.info(
                "Ignored %s below threshold %f", column_txt, remove_threshold
            )
            LOGGER.debug(
                "Column(s) with NaN (non processed): %s", column_list
            )
            return dataframe
    else:
        if callable(getattr(handle_nan, "fit_transform", None)):
            # If nan has the fit_transform method, we will trust that it
            # will do its thing correctly:
            LOGGER.info('Imputed %s with given "%s".', column_txt, handle_nan)
            LOGGER.debug("Column(s) imputed: %s", column_list)
            return run_impute(dataframe, imputer=handle_nan)
        else:
            raise ValueError(
                f'Could not make use of given imputer "{handle_nan}".'
                ' Missing ".fit_transform()"!'
            )


def preprocess_variance(
    data: pd.DataFrame, threshold: float = 0.0
) -> pd.DataFrame:
    """Remove columns with low variance."""
    numeric = data.select_dtypes(include="number")
    columns_before = set(numeric.columns)

    variance = VarianceThreshold(threshold=threshold)
    variance.fit(data.select_dtypes(include="number"))
    columns_after = set(list(variance.get_feature_names_out()))

    remove = list(columns_before - columns_after)
    if len(remove) > 0:
        LOGGER.info("Removed %i column(s) with low variance.", len(remove))
        LOGGER.debug("Column(s) removed: %s", sorted(remove))
    return data.drop(columns=remove)


def preprocess_correlations(
    data: pd.DataFrame,
    threshold: float = 1.0,
    method: str = "pearson",
) -> pd.DataFrame:
    """Remove correlated columns."""
    numeric = data.select_dtypes(include="number")

    corr = numeric.corr(method=method).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    remove = [
        column for column in upper.columns if any(upper[column] > threshold)
    ]
    if len(remove) > 0:
        LOGGER.info("Removed %i correlated column(s)", len(remove))
        LOGGER.debug("Column(s) removed: %s", sorted(remove))
    return data.drop(columns=remove)


def preprocess(
    data: pd.DataFrame,
    variance_threshold: float = 0.0,
    corr_threshold: float = 1.0,
    corr_method: str = "pearson",
    handle_nan: str | type[SimpleImputer] = "remove",
    nan_remove_threshold: float = 0.20,
) -> pd.DataFrame:
    """Remove nan/inf, low variance, and correlated features.

    Parameters
    ----------
    data : object like pd.DataFrame
        The data to process.
    variance_threshold : float, optional
        Columns with a variance lower than this value will be removed.
        If this is set to negative, no columns are removed.
    corr_threshold : float, optional
        When two columns have a correlation greater than the threshold,
        one is removed. If this is set to negative, no columns are removed.
    corr_method : string, optional
        One of the methods supported by pd.DataFrame.corr for calculating
        the correlation.
    handle_nan: string or object like SimpleImputer, optional
        Defines how we should handle NaNs. See preprocess_nan for more info.
    nan_remove_threshold : float, optional
        If the fraction of NaNs in a specific column exceeds this
        threshold, the column will be removed.
    """
    # 1. nan/info
    dataframe = preprocess_nan(
        data, handle_nan=handle_nan, remove_threshold=nan_remove_threshold
    )
    # 2. Low variance
    if variance_threshold < 0:
        LOGGER.debug(
            "Skipping variance since threshold %f < 0", variance_threshold
        )
    else:
        dataframe = preprocess_variance(
            dataframe, threshold=variance_threshold
        )
    # 3. Correlated columns
    if corr_threshold < 0:
        LOGGER.debug(
            "Skipping correlations since threshold %f < 0", corr_threshold
        )
    else:
        dataframe = preprocess_correlations(
            dataframe, threshold=corr_threshold, method=corr_method
        )
    return dataframe
