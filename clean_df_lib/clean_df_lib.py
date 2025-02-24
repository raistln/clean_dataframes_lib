import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chardet
from rapidfuzz import process, fuzz
from scipy.stats.mstats import winsorize
import re
from nltk.corpus import stopwords


def detect_load_data(csv_path, encoding=None, char_num=10000):
    """Function to detect the encoding of the CSV file and load it.

    Parameters:
    - csv_path (str): Path to the CSV file.
    - encoding (str, optional): Encoding to use. If None, it will be detected automatically.
    - char_num (int, optional): Number of characters to read for encoding detection.

    Returns:
    - pd.DataFrame or None: Loaded dataframe, or None if an error occurs.
    """
    try:
        if encoding is None:
            with open(csv_path, "rb") as rawdata:
                result = chardet.detect(rawdata.read(char_num))
            encoding = result.get("encoding", "utf-8")
            print(f"Detected encoding: {encoding}")

        df = pd.read_csv(csv_path, encoding=encoding)
        return df

    except FileNotFoundError:
        print("Error: The file was not found.")
    except pd.errors.EmptyDataError:
        print("Error: The file is empty.")
    except pd.errors.ParserError:
        print("Error: The file could not be parsed.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None  # Return None if an error occurs


def first_view_data(
    df, heat=True, colors=["#000099", "#ffff00"], title="Missing Values"
):
    """Performs an initial exploration of the dataframe.

    This function normalizes column names and string values, displays
    key information, and visualizes missing data.

    Parameters:
    - df (pd.DataFrame): The dataframe to analyze.
    - heat (bool, optional): If True, displays a heatmap of missing values.
      If False, shows a bar plot of missing values.
    - colors (list, optional): Color palette for the heatmap.
    - title (str, optional): Title for the visualization.

    Returns:
    - pd.DataFrame: The cleaned dataframe.
    """

    # Normalize string columns
    df_obj = df.select_dtypes(include=["object"])
    df[df_obj.columns] = df_obj.applymap(
        lambda s: s.lower().strip() if isinstance(s, str) else s
    )

    # Normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # Display basic info
    display(df.head())
    display(df.tail())
    df.info()

    # Visualize missing values
    if heat:
        plt.figure(figsize=(10, 4))
        sns.heatmap(df.isnull(), cbar=False, cmap=sns.color_palette(colors))
        plt.xticks(rotation=70)
        plt.title(title)  # Añadir título
        plt.show()
    else:
        df.isnull().sum().plot.bar(
            figsize=(10, 4), alpha=0.75, rot=70, color="red", fontsize=12
        )
        plt.title(title)  # Añadir título
        plt.show()

    return df


def remove_invalid_data(df, nan_col_per=51, nan_row_per=51, object_threshold=3):
    """Function that drops duplicated rows, rows and columns with NaN values above the threshold,
    and removes categorical columns with too few unique values.

    Parameters:
    - df: pandas DataFrame.
    - nan_col_per: Percentage threshold for columns with NaN values to be dropped (default 51%).
    - nan_row_per: Percentage threshold for rows with NaN values to be dropped (default 51%).
    - object_threshold: Threshold for number of unique values to drop categorical columns with too few values (default 3).

    Returns:
    - df: pandas DataFrame after cleaning.
    """
    # Drop duplicate rows
    df = df.drop_duplicates()
    # Drop duplicate columns by transposing the DataFrame, dropping duplicates, and transposing back
    df = df.T.drop_duplicates().T

    # Drop rows with NaN values above the threshold percentage
    # Calculate the minimum number of non-NaN values required to keep a row
    thresh_col_drop = round(len(df) * (1 - nan_row_per / 100), 0)
    # Drop rows that don't meet the threshold
    df = df.dropna(thresh=(thresh_col_drop), axis=0)

    # Drop columns with NaN values above the threshold percentage
    # Calculate the minimum number of non-NaN values required to keep a column
    thresh_row_drop = round(len(df.columns) * (1 - nan_col_per / 100), 0)
    # Drop columns that don't meet the threshold
    df = df.dropna(thresh=(thresh_row_drop), axis=1)

    # Drop categorical (object) columns with too few unique values
    # Select columns of type 'object' and keep only those with more unique values than the threshold
    df = df.loc[:, (df.select_dtypes(include=["object"]).nunique() > object_threshold)]

    # Calculate and display the percentage of NaNs in each column
    null_percent = [
        round(100 * df[column].isnull().sum() / df.shape[0], 1) for column in df.columns
    ]
    # Display the NaN percentages in a DataFrame
    display(pd.DataFrame(np.array(null_percent), index=df.columns, columns=["NaN_%"]).T)

    return df


def search_transf_num(df):
    """This function searches for non-numeric columns in the dataframe
    and attempts to convert them to numeric. It returns the dataframe with
    the correct column types and prints out the columns that were changed.
    """

    to_num = (
        []
    )  # List to store the names of columns that were successfully converted to numeric

    # Select columns that are not numeric (excluding datetime and timedelta types)
    df_non_numeric = df.select_dtypes(exclude=[np.number, "datetime", "timedelta"])

    # Iterate over each non-numeric column
    for column in df_non_numeric.columns:
        # Clean non-numeric characters (e.g., commas, currency symbols) that could interfere with conversion
        df[column] = df[column].replace(r"[^\d.-]", "", regex=True)

        # Attempt to convert the column to a numeric type
        try:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )  # 'coerce' will convert invalid values to NaN
            # If the conversion is successful, add the column name to the list
            to_num.append(column)
        except Exception as e:
            # If an error occurs during conversion, skip to the next column
            continue

    # Print the results
    if to_num:
        print(f"The column/s changed to numeric: {', '.join(to_num)}")
    else:
        print("No columns were converted to numeric.")

    return df  # Return the dataframe with updated column types


def search_transf_cat(df, percent=5):
    """This function searches for columns with categorical data types based on
    the number of unique values in each column compared to a percentage threshold.
    It then converts them to 'category' type and prints out the columns that were changed.
    """
    # Calculate the threshold for the number of unique values
    threshold = len(df) * (percent / 100)

    # Filter non-numeric columns
    df_non_numeric = df.select_dtypes(exclude=[np.number, "datetime", "timedelta"])

    prob_cat = []

    for column in df_non_numeric.columns:
        # Check if the number of unique values is below the threshold
        if len(df[column].value_counts()) < threshold:
            # Convert to 'category' type if condition met
            df[column] = df[column].astype("category")
            prob_cat.append(column)

    print(f"The column/s changed to categoric is/are {prob_cat}")
    return df


def search_transf_date(
    df,
    date_columns=[],
    new_columns=False,
    drop_old=False,
    regex_pattern=None,
    custom_formats=None,
):
    """
    Converts specified columns to datetime and optionally creates new columns for year, month, and day.

    Parameters:
    - df: DataFrame to process.
    - date_columns: List of columns to convert to datetime.
    - new_columns: If True, creates new columns for year, month, and day.
    - drop_old: If True, drops the original date columns.
    - regex_pattern: Custom regex pattern to clean data (default is None).
    - custom_formats: List of custom datetime formats to try (default is None).
    """
    if regex_pattern is None:
        regex_pattern = (
            r"[^\w\s\d\-\:/\.]"  # Default regex to clean unwanted characters
        )

    for col in date_columns:
        # Clean the column using the provided regex pattern
        df[col] = df[col].str.replace(regex_pattern, "", regex=True)

        # Convert to datetime with error handling
        if custom_formats:
            for fmt in custom_formats:
                try:
                    df[col] = pd.to_datetime(df[col], format=fmt, errors="raise")
                    break
                except ValueError:
                    continue
            else:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = pd.to_datetime(df[col], errors="coerce")

        # Ensure the column is of type datetime64[D]
        df[col] = df[col].astype("datetime64[D]", errors="ignore")

        if new_columns:
            # Create new columns for year, month, and day based on the cleaned date column
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day

            if drop_old:
                df = df.drop([col], axis=1)

    return df


def numeric_fill_nan(df, columns=[], fill="mean"):
    """This function fills the NaN values in specified numeric columns with a specified method
    such as mean, median, or a custom value.

    Args:
        df: DataFrame to process
        columns: List of columns to fill NaN values in (if empty, all numeric columns are processed)
        fill: Method to fill NaN values. Options are 'mean', 'median', or a custom value. Default is 'mean'.

    Returns:
        Updated DataFrame with NaN values filled.
    """
    # If no columns are specified, fill NaNs in all numeric columns
    if not columns:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    # Validate that all specified columns exist in the DataFrame
    columns = [col for col in columns if col in df.columns]

    # Loop over the selected columns
    for column in columns:
        if df[column].dtype in [
            np.number,
            "float64",
            "int64",
        ]:  # Ensure the column is numeric
            if fill == "median":
                df[column].fillna(df[column].median(), inplace=True)
            elif fill == "mean":
                df[column].fillna(df[column].mean(), inplace=True)
            else:
                df[column].fillna(fill, inplace=True)
        else:
            print(f"Warning: Column '{column}' is not numeric and will be skipped.")

    return df


def categoric_fill_nan(df, columns=None):
    """Fills the categorical columns with the mode value (most frequent) for each column in the provided list."""

    if columns is None:
        columns = df.select_dtypes(include=["object", "category"]).columns

    for column in columns:
        if column in df.columns:
            # Check if the column contains NaN values
            if df[column].isnull().any():
                # Get the mode (most frequent value)
                mode = df[column].mode()[0]
                # Fill NaN values with the mode
                df[column] = df[column].fillna(mode)

            # Convert the column to categorical type (if not already)
            if not pd.api.types.is_categorical_dtype(df[column]):
                df[column] = df[column].astype("category")

    return df


def find_matches(df, column, string_to_match, min_ratio=90, limit=10):
    """
    Finds approximate matches of a given string in a specified column.
    Returns a list of matched strings that meet or exceed the similarity threshold.
    """
    unique_values = df[column].dropna().unique()
    matches = process.extract(
        string_to_match, unique_values, limit=limit, scorer=fuzz.token_sort_ratio
    )
    close_matches = [match[0] for match in matches if match[1] >= min_ratio]
    return close_matches


def replace_matches(df, column, string_to_match, min_ratio=90, limit=10):
    """
    Replaces values in a column based on approximate string matches exceeding a given similarity ratio.
    """
    close_matches = find_matches(df, column, string_to_match, min_ratio, limit)
    df[column] = df[column].replace(close_matches, string_to_match)
    return df


def handle_high_cardinality(df, threshold=50):
    high_card_cols = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        if df[col].nunique() > threshold:
            high_card_cols.append(col)
            # Agrupar valores poco frecuentes en "otros"
            freq = df[col].value_counts(normalize=True)
            df[col] = df[col].apply(lambda x: x if freq[x] > 0.01 else "otros")
    return df, high_card_cols


def clean_column_values(
    df, column, value_type="categorical", string_to_match=None, min_ratio=90, limit=10
):
    """General function to clean values in a column depending on the value_type (categorical, numeric, etc.)."""
    if value_type == "categorical":
        # Check and fix categorical inconsistencies
        print(f"Cleaning categorical values in {column}...")
        cat_counts = categoric_incosistent_wrang(df, column)
        # Optional: Add logic to clean values based on fuzzy matching here
        # Example: Replace similar values using find_matches and replace_matches
        if string_to_match:
            df = replace_matches(df, column, string_to_match, min_ratio, limit)

    elif value_type == "numeric" and string_to_match:
        # Perform string matching and replacements if needed
        df = replace_matches(df, column, string_to_match, min_ratio, limit)

    elif value_type == "date":
        # Add date-related cleaning logic (if needed)
        pass

    return df


def plot_distribution(df, kind="hist", bins=30, figsize=(15, 10)):
    """
    Visualizes the distribution of all numeric columns in subplots.

    Parameters:
    - df: pandas DataFrame.
    - kind: Type of plot ("hist" for histogram, "kde" for density plot).
    - bins: Number of bins for the histogram (only applies if kind="hist").
    - figsize: Size of the figure.
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if not numeric_cols.any():
        print("No numeric columns to plot.")
        return

    # Calculate the number of rows and columns for subplots
    n_cols = 3  # Number of subplot columns per row
    n_rows = (len(numeric_cols) // n_cols) + 1

    # Create the figure and subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array for easier access

    # Plot each numeric column
    for i, col in enumerate(numeric_cols):
        if kind == "hist":
            sns.histplot(df[col], bins=bins, kde=True, ax=axes[i])
        elif kind == "kde":
            sns.kdeplot(df[col], ax=axes[i])
        axes[i].set_title(f"Distribution of {col}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("")

    # Hide empty axes if there are more subplots than columns
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def detect_outliers(df, threshold=1.5, plot=True):
    """
    Detects outliers in all numeric columns using the IQR method and optionally plots them.

    Parameters:
    - df: pandas DataFrame.
    - threshold: Threshold for outlier calculation (default is 1.5).
    - plot: If True, generates a boxplot highlighting the outliers.

    Returns:
    - DataFrame with the detected outliers.
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if not numeric_cols.any():
        print("No numeric columns to analyze.")
        return None

    outliers_list = []

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Filter outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        if not outliers.empty:
            outliers["outlier_column"] = (
                col  # Add a column to identify the source of the outlier
            )
            outliers_list.append(outliers)

    if outliers_list:
        outliers_df = pd.concat(outliers_list).drop_duplicates()

        # Plot outliers if requested
        if plot:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=df[numeric_cols], orient="h", palette="Set2")
            plt.title("Boxplot of Numeric Columns with Outliers Highlighted")
            plt.xlabel("Value")
            plt.ylabel("Column")

            # Highlight outliers
            for col in numeric_cols:
                col_outliers = outliers_df[outliers_df["outlier_column"] == col]
                if not col_outliers.empty:
                    plt.scatter(
                        col_outliers[col],
                        [col] * len(col_outliers),
                        color="red",
                        label="Outliers",
                    )

            # Avoid duplicate labels in the legend
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())

            plt.show()

        return outliers_df
    else:
        print("No outliers detected.")
        return None


def plot_correlation_matrix(df):
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()


def clean_text_column(df, column):
    stop_words = set(stopwords.words("english"))
    df[column] = df[column].apply(lambda x: re.sub(r"[^\w\s]", "", str(x).lower()))
    df[column] = df[column].apply(
        lambda x: " ".join([word for word in x.split() if word not in stop_words])
    )
    return df


def export_clean_data(df, path, format="csv"):
    if format == "csv":
        df.to_csv(path, index=False)
    elif format == "excel":
        df.to_excel(path, index=False)
    print(f"Data exported successfully to {path}")


def categoric_incosistent_wrang(df, column):
    """Returns a sorted list of unique values in a column, first alphabetically,
    then by the frequency of occurrences."""
    cat_counts = df[column].value_counts().to_dict()
    sorted_values = sorted(cat_counts.items(), key=lambda x: (x[0], x[1]))
    return sorted_values


def handle_outliers(df, outliers, action="remove", **kwargs):
    """
    Handles outliers in a DataFrame based on the specified action.

    Parameters:
    - df: Original DataFrame.
    - outliers: DataFrame with detected outliers (from detect_outliers).
    - action: Action to perform. Options:
        - "remove": Remove outliers from the DataFrame (default).
        - "impute": Impute outliers with median, mean, or a custom value.
        - "transform": Transform outliers using log scaling or winsorization.
        - "flag": Add a column indicating if a row is an outlier.
        - "segment": Split the DataFrame into two: one with outliers and one without.
    - **kwargs: Additional arguments depending on the action:
        - For "impute":
            - method: Imputation method ("median", "mean", or a custom value).
        - For "transform":
            - method: Transformation method ("log" for log scaling, "winsorize" for winsorization).
            - limits: Limits for winsorization (default [0.05, 0.05]).

    Returns:
    - Depending on the action, returns the modified DataFrame, a segmented DataFrame, or None.
    """
    if outliers is None or outliers.empty:
        print("No outliers to handle.")
        return df

    if action == "remove":
        print("Removing outliers...")
        df_cleaned = df[~df.index.isin(outliers.index)]
        return df_cleaned

    elif action == "impute":
        # Default imputation method is median
        method = kwargs.get("method", "median")
        print(f"Imputing outliers using {method}...")

        for col in outliers["outlier_column"].unique():
            if method == "median":
                value = df[col].median()
            elif method == "mean":
                value = df[col].mean()
            else:
                value = kwargs.get("value")  # Custom value
                if value is None:
                    raise ValueError("You must provide a value for custom imputation.")

            df.loc[outliers.index, col] = value
        return df

    elif action == "transform":
        # Default transformation method is winsorization
        method = kwargs.get("method", "winsorize")
        print(f"Transforming outliers using {method}...")

        for col in outliers["outlier_column"].unique():
            if method == "log":
                df[col] = np.log1p(df[col])  # Log scaling
            elif method == "winsorize":
                limits = kwargs.get(
                    "limits", [0.05, 0.05]
                )  # Default winsorization limits
                df[col] = winsorize(df[col], limits=limits)
            else:
                raise ValueError(
                    "Invalid transformation method. Use 'log' or 'winsorize'."
                )
        return df

    elif action == "flag":
        print("Adding outlier flag column...")
        df["is_outlier"] = df.index.isin(outliers.index)
        return df

    elif action == "segment":
        print("Splitting DataFrame into with and without outliers...")
        df_outliers = df[df.index.isin(outliers.index)]
        df_no_outliers = df[~df.index.isin(outliers.index)]
        return df_outliers, df_no_outliers

    else:
        raise ValueError(
            "Invalid action. Use 'remove', 'impute', 'transform', 'flag', or 'segment'."
        )


if __name__ == "__main__":
    pass
