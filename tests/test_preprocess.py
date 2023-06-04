import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer

from moljourney.preprocess import (
    count_nan,
    preprocess,
    preprocess_correlations,
    preprocess_nan,
    preprocess_variance,
    remove_all_nan_columns,
    remove_all_nan_rows,
    remove_nan_threshold,
    run_impute,
)

DATA = pd.DataFrame(
    {
        "col1": [0.0, 1.0, 2.0, np.nan, 11.0, 10.0],
        "col2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
)
DATA2 = pd.DataFrame(
    {
        "col1": [0.0, 1.0, 2.0, np.nan, np.inf, 10.0],
        "col2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
)
DATA3 = pd.DataFrame(
    {
        "col1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "col2": [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
    }
)
DATA4 = pd.DataFrame(
    {
        "col1": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "col2": [1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
    }
)


def test_count_nan():
    """Test that we can count nan's."""
    stat = count_nan(DATA)
    assert stat[stat["feature"] == "col1"]["nan-count"].values[0] == 1
    assert stat[stat["feature"] == "col2"]["nan-count"].values[0] == 0


def test_remove_nan_columns():
    """Test that we remove columns with nan's."""
    data = remove_all_nan_columns(DATA)
    assert "col1" not in data
    assert "col1" in DATA
    assert "col2" in data
    assert "col2" in DATA


def test_remove_nan_rows():
    """Test that we remove rows with nan's."""
    data = remove_all_nan_rows(DATA)
    assert data.shape == (5, 2)


def test_remove_nan_threshold():
    """Test that we can remove columns given a threshold for nan's."""
    data = remove_nan_threshold(DATA, threshold=0.2)
    assert data.shape == DATA.shape
    assert "col1" in data
    assert "col2" in data
    data = remove_nan_threshold(DATA, threshold=0.1)
    assert data.shape != DATA.shape
    assert "col1" not in data
    assert "col2" in data


def test_run_impute():
    """Test that we can run imputation of missing values."""
    # Test that we fail with missing parameters.
    with pytest.raises(ValueError):
        run_impute(DATA)
    # Test that we actually impute:
    assert any(DATA["col1"].isnull().values)
    data = run_impute(DATA, imputer_str="impute-median")
    assert not any(data["col1"].isnull().values)
    # Test that we can give a imputer:
    imputer = SimpleImputer(
        missing_values=np.nan, strategy="constant", fill_value=1234
    )
    data = run_impute(DATA, imputer=imputer)
    assert np.isclose(data.iloc[3, 0], 1234)


def test_preprocess_nan():
    """Test that we can preprocess nan's."""
    # This should fail since we have a inf:
    with pytest.raises(ValueError):
        preprocess_nan(DATA2, handle_nan="impute-median", include_inf=False)
    # Add handling of inf as well:
    data = preprocess_nan(
        DATA2,
        handle_nan="impute-median",
        include_inf=True,
        remove_threshold=0.5,
        force_numeric=True,
    )
    assert "col1" in data
    assert "col2" in data
    assert data.isnull().sum()["col1"] == 0
    assert data.isnull().sum()["col2"] == 0
    # Just remove missing:
    data = preprocess_nan(
        DATA2,
        handle_nan="remove",
    )
    assert "col1" not in data
    assert "col2" in data
    # Just remove samples:
    data = preprocess_nan(
        DATA2,
        handle_nan="remove-samples",
        include_inf=False,
    )
    assert "col1" in data
    assert "col2" in data
    assert data.shape == (5, 2)
    # Check that we stop after removing nan's:
    # Note, if the code above moves on to imputing, it
    # will fail due to the inf:
    data = preprocess_nan(
        DATA2,
        handle_nan="impute-median",
        remove_threshold=0.0,
    )
    assert data.shape == (6, 1)
    # Check what happens when we ask for something that is not implemented:
    data = preprocess_nan(
        DATA2,
        handle_nan="something",
        remove_threshold=0.8,
    )
    assert data.shape == DATA2.shape
    # Check that we can provinde a imputer:
    imputer = SimpleImputer(
        missing_values=np.nan, strategy="constant", fill_value=1234
    )
    data = preprocess_nan(
        DATA2,
        handle_nan=imputer,
        remove_threshold=0.8,
    )
    assert data.shape == DATA2.shape
    # Check that we fail if we provide a method that we can't use:
    with pytest.raises(ValueError):
        preprocess_nan(
            DATA2,
            handle_nan=np.argmax,
            remove_threshold=0.8,
        )


def test_preprocess_variance():
    # Test that we remove nothing:
    data = preprocess_variance(DATA3, threshold=0.0)
    assert np.allclose(data.to_numpy(), DATA3.to_numpy())
    # Test that we can remove:
    data = preprocess_variance(DATA3, threshold=0.5)
    assert "col1" in data
    assert "col2" not in data


def test_preprocess_correlations():
    # Test that we remove nothing:
    data = preprocess_correlations(DATA4, threshold=1.0)
    assert "col1" in data
    assert "col2" in data
    # Test that we remove when we are above the threshold:
    data = preprocess_correlations(DATA4, threshold=0.95)
    assert ("col1" in data) != ("col2" in data)


def test_preprocess():
    # Just test that we can run the method:
    data = preprocess(
        DATA2,
        variance_threshold=0.01,
        corr_threshold=0.95,
        handle_nan="impute-median",
    )
    assert "col2" in data
    assert "col1" not in data
    data = preprocess(
        DATA2, variance_threshold=-1, corr_threshold=-1, handle_nan="remove"
    )
    assert "col2" in data
    assert "col1" not in data
