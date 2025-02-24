import pytest
import pandas as pd
import numpy as np
import chardet
from rapidfuzz import process, fuzz
from scipy.stats.mstats import winsorize
import re
from nltk.corpus import stopwords
from clean_df_lib import (
    detect_load_data,
    first_view_data,
    remove_invalid_data,
    search_transf_cat,
    search_transf_num,
    search_transf_date,
    numeric_fill_nan,
    categoric_fill_nan,
    detect_outliers,
    handle_outliers,
    plot_distribution,
    export_clean_data
)

# Pruebas para detect_load_data
def test_detect_load_data_normal():
    df = detect_load_data("tests/test_data.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

def test_detect_load_data_empty_file():
    with pytest.raises(pd.errors.EmptyDataError):
        detect_load_data("tests/empty.csv")

def test_detect_load_data_invalid_path():
    with pytest.raises(FileNotFoundError):
        detect_load_data("tests/non_existent.csv")

def test_detect_load_data_encoding_detection():
    df = detect_load_data("tests/test_data_utf16.csv", encoding=None)
    assert isinstance(df, pd.DataFrame)

def test_detect_load_data_custom_encoding():
    df = detect_load_data("tests/test_data_utf16.csv", encoding="utf-16")
    assert isinstance(df, pd.DataFrame)

# Pruebas para first_view_data
def test_first_view_data_normal():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = first_view_data(df)
    assert isinstance(result, pd.DataFrame)

def test_first_view_data_empty():
    df = pd.DataFrame()
    result = first_view_data(df)
    assert isinstance(result, pd.DataFrame)

def test_first_view_data_missing_values():
    df = pd.DataFrame({"A": [1, None, 3], "B": [None, 5, 6]})
    result = first_view_data(df, heat=False)
    assert isinstance(result, pd.DataFrame)

def test_first_view_data_large_dataframe():
    df = pd.DataFrame({"A": range(1000), "B": range(1000)})
    result = first_view_data(df)
    assert isinstance(result, pd.DataFrame)

def test_first_view_data_custom_colors():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = first_view_data(df, colors=["#FF0000", "#00FF00"])
    assert isinstance(result, pd.DataFrame)

# Pruebas para remove_invalid_data
def test_remove_invalid_data_normal():
    df = pd.DataFrame({"A": [1, 2, None], "B": [None, None, None]})
    result = remove_invalid_data(df, nan_row_per=50)
    assert len(result) == 2

def test_remove_invalid_data_empty():
    df = pd.DataFrame()
    result = remove_invalid_data(df)
    assert result.empty

def test_remove_invalid_data_no_invalid():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = remove_invalid_data(df)
    assert len(result) == 3

def test_remove_invalid_data_all_invalid():
    df = pd.DataFrame({"A": [None, None, None], "B": [None, None, None]})
    result = remove_invalid_data(df, nan_row_per=50)
    assert result.empty

def test_remove_invalid_data_custom_threshold():
    df = pd.DataFrame({"A": [1, None, None], "B": [None, None, None]})
    result = remove_invalid_data(df, nan_row_per=66)
    assert len(result) == 1

# Pruebas para search_transf_cat
def test_search_transf_cat_normal():
    df = pd.DataFrame({"A": ["a", "b", "a"], "B": ["c", "c", "d"]})
    result = search_transf_cat(df, percent=50)
    assert pd.api.types.is_categorical_dtype(result["A"])

def test_search_transf_cat_no_categorical():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = search_transf_cat(df)
    assert not pd.api.types.is_categorical_dtype(result["A"])

def test_search_transf_cat_empty():
    df = pd.DataFrame()
    result = search_transf_cat(df)
    assert result.empty

def test_search_transf_cat_custom_threshold():
    df = pd.DataFrame({"A": ["a", "b", "a"], "B": ["c", "c", "d"]})
    result = search_transf_cat(df, percent=30)
    assert pd.api.types.is_categorical_dtype(result["A"])

def test_search_transf_cat_mixed_types():
    df = pd.DataFrame({"A": ["a", "b", "a"], "B": [1, 2, 3]})
    result = search_transf_cat(df)
    assert pd.api.types.is_categorical_dtype(result["A"])
    assert not pd.api.types.is_categorical_dtype(result["B"])

# Pruebas para detect_outliers
def test_detect_outliers_normal():
    df = pd.DataFrame({"A": [1, 2, 3, 100]})
    outliers = detect_outliers(df, threshold=1.5, plot=False)
    assert not outliers.empty

def test_detect_outliers_no_outliers():
    df = pd.DataFrame({"A": [1, 2, 3, 4]})
    outliers = detect_outliers(df, threshold=1.5, plot=False)
    assert outliers is None or outliers.empty

def test_detect_outliers_empty():
    df = pd.DataFrame()
    outliers = detect_outliers(df, threshold=1.5, plot=False)
    assert outliers is None or outliers.empty

def test_detect_outliers_custom_threshold():
    df = pd.DataFrame({"A": [1, 2, 3, 100]})
    outliers = detect_outliers(df, threshold=2.0, plot=False)
    assert not outliers.empty

def test_detect_outliers_multiple_columns():
    df = pd.DataFrame({"A": [1, 2, 3, 100], "B": [1, 2, 3, 4]})
    outliers = detect_outliers(df, threshold=1.5, plot=False)
    assert not outliers.empty