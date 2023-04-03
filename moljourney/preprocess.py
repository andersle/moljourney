"""Methods for preprocessing (nan, variance, correlations)."""
import logging

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer

logger = logging.getLogger(__name__)


def run_impute(
    dataframe: pd.DataFrame,
    imputer_str: str = "",
    imputer : type[SimpleImputer] | None = None,
) -> pd.DataFrame:
    """Run imputation on numeric part of data frame.

    Parameters
    ----------
    dataframe : object like pd.DataFrame
        The data to impute. Note: We only select the
        numeric part of the data here.
    imputer_str : string, optional
        String used to select a predefined imputer. This is
        probably not what you want, but it is a quick way
        to impute (if you just want to test imputation).
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
                f'Unknow imputer "{imputer_str}", expected one of '
                 '{list(imputers.keys())}'
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
    logger.info("Removed %i column(s) with NaNs.", len(columns_nan))
    logger.debug("Columns removed: %s", sorted(columns_nan))
    return data.dropna(axis=1)



def remove_nan_threshold(
    data: pd.DataFrame, threshold: float=0.2
) -> pd.DataFrame:
    """Remove columns where the fraction of NaN > threshold."""
    nan_fraction = data.isnull().mean()
    remove = set(nan_fraction[nan_fraction > threshold].index)
    # Remove above the given threshold:
    if len(remove) > 0:
        column_list = f"{sorted(remove)}"
        column_txt = f"{len(remove)} column(s)"
        logger.info(
            "Removed %s with NaN fraction > %f.", column_txt, threshold
        )
        logger.debug("Column(s) removed: %s", column_list)
        return data.drop(columns=remove)
    return data


def preprocess_nan(
    data: pd.DataFrame,
    handle_nan: str | type[SimpleImputer] = "remove",
    remove_threshold: float = 0.20,
    force_numeric: bool = False,
    include_inf: bool = True,
) -> pd.DataFrame:
    """Remove or impute NaNs.

    If handle_nan is remove, columns with NaNs are removed. Imputation
    can be carried out by setting this string to "impute-mean",
    "impute-median" or "impute-knn". This is probably not what you want,
    but it is included as a "quick" way to see the effect of imputation.
    If you pass an object, it will be used for imputation as long as
    it has a method "fit_transform". The remove_threshold parameter
    is used to remove columns based on the fraction of NaNs in the column.

    Parameters
    ----------
    data : object like pd.DataFrame
        The data we will process
    handle_nan : string or object like SimpleImputer.
        This is how we handle NaN's for columns where the fraction
        is missing numbers is smaller than remove_threshold.
        If a string is
        given, this is interpreted as one of the pre-defined
        methods given in ...
    remove_threshold : float, optional
        Columns with a fraction of missing numbers larger than this
        threshold will be removed.
    force_numeric : bool, optional
        If True, this method will transform all columns to numeric
        via pd.to_numeric.
    include_inf : bool, optional
        If True, this method will interpret +/- np.inf as a NaN.

    Note
    ----
    This method will make a copy of the input dataframe.
    """
    dataframe = data.copy()
    if force_numeric:
        dataframe = dataframe.apply(pd.to_numeric, errors="coerce", axis=1)
    if include_inf:
        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)

    # If handle_nan is remove, we will just remove irrespective of
    # the threshold.
    if isinstance(handle_nan, str) and handle_nan == "remove":
        return remove_all_nan_columns(dataframe)
    # Remove based on fraction of NaNs:
    dataframe = remove_nan_threshold(dataframe, threshold=remove_threshold)

    # Check if we still have some NaNs to impute:
    nan_count = dataframe.isnull().sum()
    columns_nan = set(nan_count[nan_count > 0].index)

    if len(columns_nan) == 0:
        return dataframe

    column_list = f"{sorted(columns_nan)}"
    column_txt = f"{len(columns_nan)} column(s)"

    if isinstance(handle_nan, str):
        if handle_nan.startswith("impute"):
            logger.info(
                'Imputed %s with predefined "%s" method.',
                column_txt, handle_nan
            )
            logger.debug("Column(s) imputed: %s", column_list)
            return run_impute(dataframe, imputer_str=handle_nan)
        else:
            logger.info(
                "Ignored %s below threshold %f", column_txt, remove_threshold)
            logger.debug(
                "Column(s) with NaN (non processed): %s", column_list
            )
            return dataframe
    else:
        if callable(getattr(handle_nan, "fit_transform", None)):
            # If nan has the fit_transform method, we will just trust that it
            # will do its thing correctly:
            logger.info('Imputed %s with given "%s".', column_txt, handle_nan)
            logger.debug("Column(s) imputed: %s", column_list)
            return run_impute(dataframe, imputer=handle_nan)
        else:
            raise ValueError(
                f'Could not make use of given imputer "{handle_nan}".'
                ' Missing ".fit_transform()"!'
            )


def preprocess_variance(
    data: pd.DataFrame,
    threshold: float = 0.0
) -> pd.DataFrame:
    """Remove columns with low variance."""
    numeric = data.select_dtypes(include="number")
    columns_before = set(numeric.columns)

    variance = VarianceThreshold(threshold=threshold)
    variance.fit(data.select_dtypes(include="number"))
    columns_after = set(list(variance.get_feature_names_out()))

    remove = columns_before - columns_after
    if len(remove) > 0:
        logger.info("Removed %i column(s) with low variance.", len(remove))
        logger.debug("Column(s) removed: %s", sorted(remove))
    return data.drop(columns=remove)


def preprocess_correlations(
    data: pd.DataFrame,
    threshold: float = 1.0
) -> pd.DataFrame:
    """Remove correlated columns."""
    numeric = data.select_dtypes(include="number")

    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    remove = [
        column for column in upper.columns if any(upper[column] > threshold)
    ]
    if len(remove) > 0:
        logger.info("Removed %i correlated column(s)", len(remove))
        logger.debug("Column(s) removed: %s", sorted(remove))
    return data.drop(columns=remove)


def preprocess(
    data: pd.DataFrame,
    variance_threshold: float = 0.0,
    corr_threshold: float = 1.0,
    handle_nan: str | type[SimpleImputer] = "remove",
    nan_remove_threshold: float = 0.20,
)-> pd.DataFrame:
    """Remove nan/inf, low variance, and correlated features.

    Parameters
    ----------
    data : object like pd.DataFrame
        The data to process.
    variance_threshold : float, optional
        Columns with a variance lower than this value will be removed.
        If this is set to negative, no columns are removed.
    corr_threshold : float, optional
        Columns with correlation larger than this value will be removed.
        When two columns have a correlation larger than the threshold, one of
        them is removed. If this is set to negative, no columns are removed.
    handle_nan: string or object like SimpleImputer, optional
        Defines how we should handle NaNs. See preprocess_nan for more iunfo.
    nan_remove_threshold : float, optional
        If the fraction of NaNs in a specific column is higher than this
        threshold, the columns will be removed.
    """
    # 1. nan/info
    dataframe = preprocess_nan(
        data, handle_nan=handle_nan, remove_threshold=nan_remove_threshold
    )
    # 2. Low variance
    if variance_threshold < 0:
        logger.debug(
            "Skipping variance since threshold %f < 0", variance_threshold
        )
    else:
        dataframe = preprocess_variance(
            dataframe, threshold=variance_threshold
        )
    # 3. Correlated columns
    if corr_threshold < 0:
        logger.debug(
            "Skipping correlations since threshold %f < 0", corr_threshold
        )
    else:
        dataframe = preprocess_correlations(
            dataframe, threshold=corr_threshold
        )
    return dataframe
