# Data Cleaning Toolkit

This repository contains a collection of functions designed to facilitate the process of loading, cleaning, and transforming data in Python, especially focused on Exploratory Data Analysis (EDA).

## License

This project is licensed under the GNU General Public License v3 (GPLv3). See the [LICENSE](LICENSE) file for details.

## Features

- Automatic detection of CSV file encoding
- Initial data exploration with visualization of missing values
- Cleaning and transformation of different data types (numeric, categorical, dates)
- Detection and handling of outliers
- Functions for visualizing distributions and correlations
- Tools for handling inconsistencies in categorical data
- Export of cleaned data in different formats

## Requirements
numpy
pandas
matplotlib
seaborn
chardet
rapidfuzz
scipy
nltk
## Main Functions

### Data Loading

- `detect_load_data(csv_path, encoding=None, char_num=10000)`: Automatically detects the encoding of the CSV file and loads it into a DataFrame.

### Initial Exploration

- `first_view_data(df, heat=True, colors=["#000099", "#ffff00"], title="Missing Values")`: Performs an initial exploration of the DataFrame, normalizes column names and values, and visualizes missing data.

- `remove_invalid_data(df, nan_col_per=51, nan_row_per=51, object_threshold=3)`: Removes duplicate rows, columns with too many NaN values, and categorical columns with very few unique values.

### Data Type Transformation

- `search_transf_num(df)`: Identifies and converts non-numeric columns to numeric format when possible.

- `search_transf_cat(df, percent=5)`: Identifies and converts potentially categorical columns based on the number of unique values.

- `search_transf_date(df, date_columns=[], new_columns=False, drop_old=False, regex_pattern=None, custom_formats=None)`: Converts specific columns to date format and optionally creates columns for year, month, and day.

### Handling Missing Values

- `numeric_fill_nan(df, columns=[], fill="mean")`: Fills NaN values in numeric columns using the mean, median, or a custom value.

- `categoric_fill_nan(df, columns=None)`: Fills NaN values in categorical columns with the most frequent value (mode).

### Handling Inconsistencies

- `find_matches(df, column, string_to_match, min_ratio=90, limit=10)`: Finds approximate matches of a string in a specific column.

- `replace_matches(df, column, string_to_match, min_ratio=90, limit=10)`: Replaces values in a column based on approximate matches.

- `handle_high_cardinality(df, threshold=50)`: Handles categorical columns with high cardinality by grouping infrequent values.

- `clean_column_values(df, column, value_type="categorical", string_to_match=None, min_ratio=90, limit=10)`: General function to clean values in a column depending on the value type.

- `categoric_incosistent_wrang(df, column)`: Returns a sorted list of unique values in a column, first alphabetically and then by frequency.

### Visualization

- `plot_distribution(df, kind="hist", bins=30, figsize=(15, 10))`: Visualizes the distribution of all numeric columns in subplots.

- `plot_correlation_matrix(df)`: Visualizes the correlation matrix of numeric columns.

### Detection and Handling of Outliers

- `detect_outliers(df, threshold=1.5, plot=True)`: Detects outliers in numeric columns using the IQR method.

- `handle_outliers(df, outliers, action="remove", **kwargs)`: Handles outliers in a DataFrame based on the specified action (remove, impute, transform, flag, or segment).

### Text Cleaning and Export

- `clean_text_column(df, column)`: Cleans text columns by removing punctuation, converting to lowercase, and removing stop words.

- `export_clean_data(df, path, format="csv")`: Exports the cleaned DataFrame to a file in the specified format.

## Usage Example

```python
# Import the module
import data_cleaning_toolkit as dct

# Load data with automatic encoding detection
df = dct.detect_load_data("my_file.csv")

# Initial exploration and visualization of missing data
df = dct.first_view_data(df)

# Remove invalid data
df = dct.remove_invalid_data(df)

# Convert columns to their appropriate types
df = dct.search_transf_num(df)
df = dct.search_transf_cat(df)
df = dct.search_transf_date(df, date_columns=["sale_date"], new_columns=True)

# Fill missing values
df = dct.numeric_fill_nan(df)
df = dct.categoric_fill_nan(df)

# Detect and handle outliers
outliers = dct.detect_outliers(df)
df = dct.handle_outliers(df, outliers, action="transform", method="winsorize")

# Visualize distributions
dct.plot_distribution(df)
dct.plot_correlation_matrix(df)

# Export clean data
dct.export_clean_data(df, "clean_data.csv")

## Contributions
Contributions are welcome. Please open an issue or a pull request for suggestions or improvements.