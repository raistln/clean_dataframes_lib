# clean_df_lib/__init__.py
"""
Clean DataFrames Library (clean_df_lib)

A Python package for cleaning and analyzing data in Pandas DataFrames.
Provides functions for handling missing values, detecting outliers, and more.
"""

import os  

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

from .clean_df_lib import (
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

__all__ = [
    "detect_load_data",
    "first_view_data",
    "remove_invalid_data",
    "search_transf_cat",
    "search_transf_num",
    "search_transf_date",
    "numeric_fill_nan",
    "categoric_fill_nan",
    "detect_outliers",
    "handle_outliers",
    "plot_distribution",
    "export_clean_data",
    "find_matches",
    "replace_matches",
    "handle_high_cardinality",
    "clean_column_values",
    "plot_correlation_matrix",
    "clean_text_column",
    "categoric_inconsistent_wrang"
]