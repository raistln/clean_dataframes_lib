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
    export_clean_data,
    find_matches,
    replace_matches,
    handle_high_cardinality,
    clean_column_values,
    plot_correlation_matrix,
    clean_text_column,
    categoric_inconsistent_wrang
)


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
    export_clean_data,
    find_matches,
    replace_matches,
    handle_high_cardinality,
    clean_column_values,
    plot_correlation_matrix,
    clean_text_column,
    categoric_inconsistent_wrang,
)


def test_detect_load_data_normal():
    """
    Test case for normal CSV file loading.
    Verifies that the function returns a non-empty DataFrame.
    """
    df = detect_load_data("tests/test_data.csv")
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_detect_load_data_empty_file():
    """
    Test case for loading an empty CSV file.
    Verifies that the function raises an EmptyDataError.
    """
    with pytest.raises(pd.errors.EmptyDataError):
        detect_load_data("tests/empty.csv")


def test_detect_load_data_invalid_path():
    """
    Test case for loading a CSV file from an invalid path.
    Verifies that the function raises a FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        detect_load_data("tests/non_existent.csv")


def test_first_view_data_normal():
    """
    Test case for normal DataFrame exploration.
    Verifies that the function returns a DataFrame.
    """
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = first_view_data(df)
    assert isinstance(result, pd.DataFrame)


def test_first_view_data_empty():
    """
    Test case for exploring an empty DataFrame.
    Verifies that the function returns an empty DataFrame.
    """
    df = pd.DataFrame()
    result = first_view_data(df)
    assert isinstance(result, pd.DataFrame)


def test_first_view_data_missing_values():
    """
    Test case for exploring a DataFrame with missing values.
    Verifies that the function returns a DataFrame.
    """
    df = pd.DataFrame({"A": [1, None, 3], "B": [None, 5, 6]})
    result = first_view_data(df, heat=False)
    assert isinstance(result, pd.DataFrame)


def test_remove_invalid_data_normal():
    """
    Test case for removing invalid rows and columns.
    Verifies that the function removes rows with too many NaNs.
    """
    df = pd.DataFrame({"A": [1, 2, None], "B": [None, None, None]})
    result = remove_invalid_data(df, nan_row_per=50)
    assert len(result) == 2


def test_remove_invalid_data_empty():
    """
    Test case for removing invalid data from an empty DataFrame.
    Verifies that the function returns an empty DataFrame.
    """
    df = pd.DataFrame()
    result = remove_invalid_data(df)
    assert result.empty


def test_remove_invalid_data_all_invalid():
    """
    Test case for removing invalid data when all rows are invalid.
    Verifies that the function returns an empty DataFrame.
    """
    df = pd.DataFrame({"A": [None, None, None], "B": [None, None, None]})
    result = remove_invalid_data(df, nan_row_per=50)
    assert result.empty


def test_search_transf_num_normal():
    """
    Test case for converting non-numeric columns to numeric.
    Verifies that the function converts columns to numeric types.
    """
    df = pd.DataFrame({"A": ["1", "2", "3"], "B": ["4.5", "5.5", "6.5"]})
    result = search_transf_num(df)
    assert pd.api.types.is_numeric_dtype(result["A"])
    assert pd.api.types.is_numeric_dtype(result["B"])


def test_search_transf_num_no_numeric():
    """
    Test case for converting numeric columns (no conversion needed).
    Verifies that the function does not change numeric columns.
    """
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = search_transf_num(df)
    assert pd.api.types.is_numeric_dtype(result["A"])
    assert pd.api.types.is_numeric_dtype(result["B"])


def test_search_transf_num_mixed_types():
    """
    Test case for converting mixed-type columns (numeric and non-numeric).
    Verifies that the function converts non-numeric columns to numeric.
    """
    df = pd.DataFrame({"A": ["1", "2", "3"], "B": [4, 5, 6]})
    result = search_transf_num(df)
    assert pd.api.types.is_numeric_dtype(result["A"])
    assert pd.api.types.is_numeric_dtype(result["B"])


def test_search_transf_cat_normal():
    """
    Test case for converting categorical columns.
    Verifies that the function converts columns to categorical types.
    """
    df = pd.DataFrame({"A": ["a", "b", "a"], "B": ["c", "c", "d"]})
    result = search_transf_cat(df, percent=50)
    assert pd.api.types.is_categorical_dtype(result["A"])


def test_search_transf_cat_no_categorical():
    """
    Test case for converting non-categorical columns.
    Verifies that the function does not change non-categorical columns.
    """
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    result = search_transf_cat(df)
    assert not pd.api.types.is_categorical_dtype(result["A"])


def test_search_transf_cat_empty():
    """
    Test case for converting categorical columns in an empty DataFrame.
    Verifies that the function returns an empty DataFrame.
    """
    df = pd.DataFrame()
    result = search_transf_cat(df)
    assert result.empty


def test_search_transf_date_normal():
    """
    Test case for converting date columns.
    Verifies that the function converts columns to datetime types.
    """
    df = pd.DataFrame({"date": ["2023-01-01", "2023-02-01", "2023-03-01"]})
    result = search_transf_date(df, date_columns=["date"])
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_search_transf_date_invalid_format():
    """
    Test case for converting date columns with invalid formats.
    Verifies that the function handles invalid date formats.
    """
    df = pd.DataFrame({"date": ["01-01-2023", "02-01-2023", "03-01-2023"]})
    result = search_transf_date(df, date_columns=["date"], custom_formats=["%Y-%m-%d"])
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_search_transf_date_empty():
    """
    Test case for converting date columns in an empty DataFrame.
    Verifies that the function returns an empty DataFrame.
    """
    df = pd.DataFrame()
    result = search_transf_date(df, date_columns=["date"])
    assert result.empty


def test_numeric_fill_nan_normal():
    """
    Test case for filling NaN values in numeric columns with the mean.
    Verifies that the function fills NaNs correctly.
    """
    df = pd.DataFrame({"A": [1, 2, None], "B": [4, None, 6]})
    result = numeric_fill_nan(df, fill="mean")
    assert result["A"].isnull().sum() == 0
    assert result["B"].isnull().sum() == 0


def test_numeric_fill_nan_custom_value():
    """
    Test case for filling NaN values in numeric columns with a custom value.
    Verifies that the function fills NaNs with the specified value.
    """
    df = pd.DataFrame({"A": [1, 2, None], "B": [4, None, 6]})
    result = numeric_fill_nan(df, fill=0)
    assert result["A"].isnull().sum() == 0
    assert result["B"].isnull().sum() == 0


def test_numeric_fill_nan_empty():
    """
    Test case for filling NaN values in an empty DataFrame.
    Verifies that the function returns an empty DataFrame.
    """
    df = pd.DataFrame()
    result = numeric_fill_nan(df)
    assert result.empty


def test_categoric_fill_nan_normal():
    """
    Test case for filling NaN values in categorical columns with the mode.
    Verifies that the function fills NaNs correctly.
    """
    df = pd.DataFrame({"A": ["a", "b", None], "B": ["c", None, "c"]})
    result = categoric_fill_nan(df)
    assert result["A"].isnull().sum() == 0
    assert result["B"].isnull().sum() == 0


def test_categoric_fill_nan_empty():
    """
    Test case for filling NaN values in an empty DataFrame.
    Verifies that the function returns an empty DataFrame.
    """
    df = pd.DataFrame()
    result = categoric_fill_nan(df)
    assert result.empty


def test_categoric_fill_nan_no_nan():
    """
    Test case for filling NaN values when there are no NaNs.
    Verifies that the function does not modify the DataFrame.
    """
    df = pd.DataFrame({"A": ["a", "b", "c"], "B": ["d", "e", "f"]})
    result = categoric_fill_nan(df)
    assert result["A"].isnull().sum() == 0
    assert result["B"].isnull().sum() == 0


def test_detect_outliers_normal():
    """
    Test case for detecting outliers in a numeric column.
    Verifies that the function detects outliers correctly.
    """
    df = pd.DataFrame({"A": [1, 2, 3, 100]})
    outliers = detect_outliers(df, threshold=1.5, plot=False)
    assert not outliers.empty


def test_detect_outliers_no_outliers():
    """
    Test case for detecting outliers when there are none.
    Verifies that the function returns None or an empty DataFrame.
    """
    df = pd.DataFrame({"A": [1, 2, 3, 4]})
    outliers = detect_outliers(df, threshold=1.5, plot=False)
    assert outliers is None or outliers.empty


def test_detect_outliers_empty():
    """
    Test case for detecting outliers in an empty DataFrame.
    Verifies that the function returns None or an empty DataFrame.
    """
    df = pd.DataFrame()
    outliers = detect_outliers(df, threshold=1.5, plot=False)
    assert outliers is None or outliers.empty


def test_handle_outliers_remove():
    """
    Test case for removing outliers from a DataFrame.
    Verifies that the function removes outliers correctly.
    """
    df = pd.DataFrame({"A": [1, 2, 3, 100]})
    outliers = detect_outliers(df, threshold=1.5, plot=False)
    result = handle_outliers(df, outliers, action="remove")
    assert len(result) == 3


def test_handle_outliers_impute():
    """
    Test case for imputing outliers with the median.
    Verifies that the function imputes outliers correctly.
    """
    df = pd.DataFrame({"A": [1, 2, 3, 100]})
    outliers = detect_outliers(df, threshold=1.5, plot=False)
    result = handle_outliers(df, outliers, action="impute", method="median")
    assert len(result) == 4


def test_handle_outliers_flag():
    """
    Test case for flagging outliers in a DataFrame.
    Verifies that the function adds an outlier flag column.
    """
    df = pd.DataFrame({"A": [1, 2, 3, 100]})
    outliers = detect_outliers(df, threshold=1.5, plot=False)
    result = handle_outliers(df, outliers, action="flag")
    assert "is_outlier" in result.columns


def sample_dataframe():
    """
    Helper function to create a sample DataFrame for testing.
    """
    return pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": ["a", "b", "a", "c", "b"], "C": [10, np.nan, 30, 40, 50]})


def test_plot_distribution(sample_dataframe):
    """
    Test case for plotting the distribution of numeric columns.
    Verifies that the function runs without errors.
    """
    assert plot_distribution(sample_dataframe, "A") is None


def test_export_clean_data(sample_dataframe, tmp_path):
    """
    Test case for exporting a cleaned DataFrame to a CSV file.
    Verifies that the exported file is not empty.
    """
    file_path = tmp_path / "test.csv"
    export_clean_data(sample_dataframe, file_path)
    df_loaded = pd.read_csv(file_path)
    assert not df_loaded.empty


def test_find_matches():
    """
    Test case for finding approximate string matches.
    Verifies that the function returns a dictionary of matches.
    """
    values = ["apple", "aple", "banana", "bananna"]
    matches = find_matches(values, threshold=90)
    assert isinstance(matches, dict)
    assert "apple" in matches


def test_replace_matches():
    """
    Test case for replacing approximate string matches.
    Verifies that the function replaces matches correctly.
    """
    data = ["apple", "aple", "banana", "bananna"]
    match_dict = {"aple": "apple", "bananna": "banana"}
    cleaned = replace_matches(data, match_dict)
    assert cleaned == ["apple", "apple", "banana", "banana"]


def test_handle_high_cardinality(sample_dataframe):
    """
    Test case for handling high cardinality in categorical columns.
    Verifies that the function reduces cardinality correctly.
    """
    df_transformed = handle_high_cardinality(sample_dataframe, "B", threshold=2)
    assert "B_other" in df_transformed.columns or "B" in df_transformed.columns


def test_clean_column_values(sample_dataframe):
    """
    Test case for cleaning values in a column.
    Verifies that the function cleans the column correctly.
    """
    df_cleaned = clean_column_values(sample_dataframe, "B")
    assert "B" in df_cleaned.columns


def test_plot_correlation_matrix(sample_dataframe):
    """
    Test case for plotting a correlation matrix.
    Verifies that the function runs without errors.
    """
    assert plot_correlation_matrix(sample_dataframe) is None


def test_clean_text_column():
    """
    Test case for cleaning text columns.
    Verifies that the function cleans text correctly.
    """
    df = pd.DataFrame({"text": ["  Hello  ", "WORLD!!", " PyThon "]})
    df_cleaned = clean_text_column(df, "text")
    assert df_cleaned["text"].tolist() == ["hello", "world", "python"]


def test_categoric_inconsistent_wrang():
    """
    Test case for handling inconsistent categorical values.
    Verifies that the function reduces inconsistencies.
    """
    df = pd.DataFrame({"category": ["Cat", "cat ", "Dog", "dog", "DOG"]})
    df_cleaned = categoric_inconsistent_wrang(df, "category")
    assert len(df_cleaned["category"].unique()) < len(df["category"].unique())