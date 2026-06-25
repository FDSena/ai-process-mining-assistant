from pathlib import Path
import argparse
import json
import pandas as pd


PROCESSED_DATA_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")


def select_csv_file(processed_data_dir: Path) -> Path:
    """
    Let the user select a cleaned CSV file from the processed data directory.
    """
    csv_files = sorted(processed_data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {processed_data_dir}. "
            "Please run data_preprocessing.py first."
        )

    print("\nAvailable CSV files in data/processed:\n")

    for index, file_path in enumerate(csv_files, start=1):
        print(f"[{index}] {file_path.name}")

    while True:
        choice = input("\nSelect a file number: ")

        if choice.isdigit():
            choice_index = int(choice)

            if 1 <= choice_index <= len(csv_files):
                return csv_files[choice_index - 1]

        print("Invalid choice. Please enter a valid number.")


def build_default_output_path(input_path: Path) -> Path:
    """
    Build the default JSON report path.

    Example:
    data/processed/sales_data_clean.csv
    becomes:
    reports/sales_data_profile.json
    """
    filename = input_path.stem

    if filename.endswith("_clean"):
        filename = filename.replace("_clean", "")

    output_filename = f"{filename}_profile.json"

    return REPORTS_DIR / output_filename


def load_data(input_path: Path) -> pd.DataFrame:
    """
    Load a cleaned CSV file.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"File not found: {input_path}. "
            "Please check the path."
        )

    return pd.read_csv(input_path)


def detect_column_types(df: pd.DataFrame) -> dict:
    """
    Detect numerical, categorical, datetime, and boolean columns.
    """
    numerical_columns = []
    categorical_columns = []
    datetime_columns = []
    boolean_columns = []

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_bool_dtype(series):
            boolean_columns.append(column)

        elif pd.api.types.is_numeric_dtype(series):
            numerical_columns.append(column)

        else:
            converted_dates = pd.to_datetime(series, errors="coerce")
            valid_dates_ratio = converted_dates.notna().mean()

            if valid_dates_ratio >= 0.8:
                datetime_columns.append(column)
            else:
                categorical_columns.append(column)

    return {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "datetime_columns": datetime_columns,
        "boolean_columns": boolean_columns,
    }


def get_missing_values(df: pd.DataFrame) -> dict:
    """
    Return missing values count and percentage for each column.
    """
    missing_summary = {}

    for column in df.columns:
        missing_count = int(df[column].isna().sum())
        missing_percentage = float(df[column].isna().mean() * 100)

        missing_summary[column] = {
            "missing_count": missing_count,
            "missing_percentage": round(missing_percentage, 2),
        }

    return missing_summary


def get_numerical_summary(df: pd.DataFrame, numerical_columns: list[str]) -> dict:
    """
    Return descriptive statistics for numerical columns.
    """
    summary = {}

    for column in numerical_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        summary[column] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
        }

    return summary


def get_categorical_summary(
    df: pd.DataFrame,
    categorical_columns: list[str],
    max_values: int = 10,
) -> dict:
    """
    Return number of unique values and top values for categorical columns.
    """
    summary = {}

    for column in categorical_columns:
        series = df[column].dropna()

        top_values = series.value_counts().head(max_values)

        summary[column] = {
            "unique_values": int(series.nunique()),
            "top_values": {
                str(index): int(value)
                for index, value in top_values.items()
            },
        }

    return summary


def get_datetime_summary(df: pd.DataFrame, datetime_columns: list[str]) -> dict:
    """
    Return min and max dates for datetime columns.
    """
    summary = {}

    for column in datetime_columns:
        converted = pd.to_datetime(df[column], errors="coerce").dropna()

        if converted.empty:
            continue

        summary[column] = {
            "min_date": str(converted.min()),
            "max_date": str(converted.max()),
        }

    return summary


def get_duplicate_summary(df: pd.DataFrame) -> dict:
    """
    Return duplicated row information.
    """
    duplicate_count = int(df.duplicated().sum())
    duplicate_percentage = float(df.duplicated().mean() * 100)

    return {
        "duplicate_rows": duplicate_count,
        "duplicate_percentage": round(duplicate_percentage, 2),
    }


def get_simple_outliers(df: pd.DataFrame, numerical_columns: list[str]) -> dict:
    """
    Detect simple outliers using the IQR method.
    """
    outliers = {}

    for column in numerical_columns:
        series = df[column].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            outliers[column] = {
                "outlier_count": 0,
                "outlier_percentage": 0.0,
            }
            continue

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_mask = (series < lower_bound) | (series > upper_bound)
        outlier_count = int(outlier_mask.sum())
        outlier_percentage = float(outlier_mask.mean() * 100)

        outliers[column] = {
            "outlier_count": outlier_count,
            "outlier_percentage": round(outlier_percentage, 2),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
        }

    return outliers


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Create a complete profile of the dataset.
    """
    column_types = detect_column_types(df)

    profile = {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": df.columns.tolist(),
        "column_types": column_types,
        "missing_values": get_missing_values(df),
        "duplicates": get_duplicate_summary(df),
        "numerical_summary": get_numerical_summary(
            df,
            column_types["numerical_columns"],
        ),
        "categorical_summary": get_categorical_summary(
            df,
            column_types["categorical_columns"],
        ),
        "datetime_summary": get_datetime_summary(
            df,
            column_types["datetime_columns"],
        ),
        "outliers": get_simple_outliers(
            df,
            column_types["numerical_columns"],
        ),
    }

    return profile


def save_profile(profile: dict, output_path: Path) -> None:
    """
    Save the dataset profile as a JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(profile, file, indent=4, ensure_ascii=False)

    print(f"\nProfile saved to: {output_path}")


def print_profile_overview(profile: dict) -> None:
    """
    Print a readable overview of the profile.
    """
    print("\nDataset profile overview")
    print("------------------------")

    print("\nShape:")
    print(profile["shape"])

    print("\nNumerical columns:")
    print(profile["column_types"]["numerical_columns"])

    print("\nCategorical columns:")
    print(profile["column_types"]["categorical_columns"])

    print("\nDatetime columns:")
    print(profile["column_types"]["datetime_columns"])

    print("\nBoolean columns:")
    print(profile["column_types"]["boolean_columns"])

    print("\nDuplicate rows:")
    print(profile["duplicates"])

    print("\nColumns with missing values:")
    columns_with_missing_values = {
        column: values
        for column, values in profile["missing_values"].items()
        if values["missing_count"] > 0
    }

    if columns_with_missing_values:
        print(columns_with_missing_values)
    else:
        print("No missing values detected.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile a cleaned CSV dataset."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to the cleaned CSV file. "
            "If not provided, the script lets you choose a file from data/processed/."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path where the JSON profile will be saved. "
            "If not provided, it is generated from the input file name."
        ),
    )

    args = parser.parse_args()

    if args.input is None:
        input_path = select_csv_file(PROCESSED_DATA_DIR)
    else:
        input_path = Path(args.input)

    if args.output is None:
        output_path = build_default_output_path(input_path)
    else:
        output_path = Path(args.output)

    print(f"\nSelected input file: {input_path}")
    print(f"Output file: {output_path}")

    df = load_data(input_path)
    profile = profile_dataset(df)

    print_profile_overview(profile)
    save_profile(profile, output_path)


if __name__ == "__main__":
    main()