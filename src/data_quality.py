from pathlib import Path
import argparse
import json
import pandas as pd
import warnings


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
    Build the default data quality report path.

    Example:
    data/processed/sales_data_clean.csv
    becomes:
    reports/sales_data_quality.json
    """
    filename = input_path.stem

    if filename.endswith("_clean"):
        filename = filename.replace("_clean", "")

    output_filename = f"{filename}_quality.json"

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


def detect_empty_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect columns where all values are missing.
    """
    return df.columns[df.isna().all()].tolist()


def detect_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = 30.0,
) -> dict:
    """
    Detect columns with a high percentage of missing values.
    """
    result = {}

    for column in df.columns:
        missing_percentage = df[column].isna().mean() * 100

        if missing_percentage >= threshold:
            result[column] = round(float(missing_percentage), 2)

    return result


def detect_constant_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect columns with only one unique non-missing value.
    """
    constant_columns = []

    for column in df.columns:
        if df[column].nunique(dropna=True) <= 1:
            constant_columns.append(column)

    return constant_columns


def detect_high_cardinality_columns(
    df: pd.DataFrame,
    date_columns: list[str],
    threshold: int = 100,
) -> dict:
    """
    Detect categorical columns with many unique values.
    """
    high_cardinality_columns = {}

    for column in df.columns:
        if column in date_columns:
            continue

        if not pd.api.types.is_numeric_dtype(df[column]):
            unique_count = int(df[column].nunique(dropna=True))

            if unique_count >= threshold:
                high_cardinality_columns[column] = unique_count

    return high_cardinality_columns


def detect_possible_id_columns(
    df: pd.DataFrame,
    date_columns: list[str],
    uniqueness_threshold: float = 0.9,
) -> dict:
    """
    Detect columns that may be identifiers because most values are unique.
    """
    possible_id_columns = {}

    row_count = len(df)

    if row_count == 0:
        return possible_id_columns

    for column in df.columns:
        if column in date_columns:
            continue

        unique_ratio = df[column].nunique(dropna=True) / row_count

        if unique_ratio >= uniqueness_threshold:
            possible_id_columns[column] = round(float(unique_ratio), 4)

    return possible_id_columns

def detect_negative_values(df: pd.DataFrame) -> dict:
    """
    Detect negative values in numerical columns.
    """
    negative_values = {}

    numerical_columns = df.select_dtypes(include="number").columns

    for column in numerical_columns:
        negative_count = int((df[column] < 0).sum())

        if negative_count > 0:
            negative_values[column] = negative_count

    return negative_values


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect columns that are likely to be dates.
    """
    date_columns = []

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            converted = pd.to_datetime(df[column], errors="coerce")

        valid_ratio = converted.notna().mean()

        if valid_ratio >= 0.8:
            date_columns.append(column)

    return date_columns

def detect_invalid_dates(df: pd.DataFrame, date_columns: list[str]) -> dict:
    """
    Detect invalid date values in detected date columns.
    """
    invalid_dates = {}

    for column in date_columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            converted = pd.to_datetime(df[column], errors="coerce")

        invalid_count = int(converted.isna().sum())

        if invalid_count > 0:
            invalid_dates[column] = invalid_count

    return invalid_dates

def generate_quality_report(df: pd.DataFrame) -> dict:
    """
    Generate a data quality report.
    """
    date_columns = detect_date_columns(df)

    report = {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "empty_columns": detect_empty_columns(df),
        "high_missing_columns": detect_high_missing_columns(df),
        "constant_columns": detect_constant_columns(df),
        "high_cardinality_columns": detect_high_cardinality_columns(df, date_columns),
        "possible_id_columns": detect_possible_id_columns(df, date_columns),
        "negative_values": detect_negative_values(df),
        "date_columns": date_columns,
        "invalid_dates": detect_invalid_dates(df, date_columns),
    }

    return report

def generate_quality_alerts(report: dict) -> list[str]:
    """
    Generate readable data quality alerts.
    """
    alerts = []

    if report["empty_columns"]:
        alerts.append(
            "Some columns are completely empty: "
            + ", ".join(report["empty_columns"])
            + "."
        )

    if report["high_missing_columns"]:
        columns = ", ".join(report["high_missing_columns"].keys())
        alerts.append(
            "Some columns have a high percentage of missing values: "
            + columns
            + "."
        )

    if report["constant_columns"]:
        alerts.append(
            "Some columns have only one unique value and may not be useful for analysis: "
            + ", ".join(report["constant_columns"])
            + "."
        )

    if report["high_cardinality_columns"]:
        columns = ", ".join(report["high_cardinality_columns"].keys())
        alerts.append(
            "Some categorical columns have many unique values and may need special handling: "
            + columns
            + "."
        )

    if report["possible_id_columns"]:
        columns = ", ".join(report["possible_id_columns"].keys())
        alerts.append(
            "Some columns look like identifier columns because most values are unique: "
            + columns
            + "."
        )

    if report["negative_values"]:
        columns = ", ".join(report["negative_values"].keys())
        alerts.append(
            "Some numerical columns contain negative values: "
            + columns
            + "."
        )

    if report["invalid_dates"]:
        columns = ", ".join(report["invalid_dates"].keys())
        alerts.append(
            "Some date columns contain invalid date values: "
            + columns
            + "."
        )

    if not alerts:
        alerts.append("No major data quality issues were detected.")

    return alerts


def save_quality_report(report: dict, output_path: Path) -> None:
    """
    Save the data quality report as JSON.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print(f"\nData quality report saved to: {output_path}")


def print_quality_report(report: dict) -> None:
    """
    Print a readable quality report.
    """
    print("\nData quality report")
    print("-------------------")

    print("\nShape:")
    print(report["shape"])

    print("\nDetected alerts:")
    for alert in report["alerts"]:
        print(f"- {alert}")

    print("\nDetails:")
    print(f"Empty columns: {report['empty_columns']}")
    print(f"High missing columns: {report['high_missing_columns']}")
    print(f"Constant columns: {report['constant_columns']}")
    print(f"High cardinality columns: {report['high_cardinality_columns']}")
    print(f"Possible ID columns: {report['possible_id_columns']}")
    print(f"Negative values: {report['negative_values']}")
    print(f"Date columns: {report['date_columns']}")
    print(f"Invalid dates: {report['invalid_dates']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run data quality checks on a cleaned CSV dataset."
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
            "Path where the data quality JSON report will be saved. "
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

    report = generate_quality_report(df)
    report["alerts"] = generate_quality_alerts(report)

    print_quality_report(report)
    save_quality_report(report, output_path)


if __name__ == "__main__":
    main()