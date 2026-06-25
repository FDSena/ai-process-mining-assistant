from pathlib import Path
import argparse
import json


REPORTS_DIR = Path("reports")


def select_profile_file(reports_dir: Path) -> Path:
    """
    Let the user select a JSON profile file from the reports directory.
    """
    profile_files = sorted(reports_dir.glob("*_profile.json"))

    if not profile_files:
        raise FileNotFoundError(
            f"No profile files found in {reports_dir}. "
            "Please run data_profiler.py first."
        )

    print("\nAvailable profile files in reports:\n")

    for index, file_path in enumerate(profile_files, start=1):
        print(f"[{index}] {file_path.name}")

    while True:
        choice = input("\nSelect a file number: ")

        if choice.isdigit():
            choice_index = int(choice)

            if 1 <= choice_index <= len(profile_files):
                return profile_files[choice_index - 1]

        print("Invalid choice. Please enter a valid number.")


def build_default_output_path(input_path: Path) -> Path:
    """
    Build the default insights JSON path.

    Example:
    reports/sales_data_profile.json
    becomes:
    reports/sales_data_insights.json
    """
    filename = input_path.stem

    if filename.endswith("_profile"):
        filename = filename.replace("_profile", "")

    output_filename = f"{filename}_insights.json"

    return REPORTS_DIR / output_filename


def load_profile(input_path: Path) -> dict:
    """
    Load the dataset profile JSON file.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"File not found: {input_path}. "
            "Please check the path or run data_profiler.py first."
        )

    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def generate_shape_insights(profile: dict) -> list[str]:
    """
    Generate insights about the dataset shape.
    """
    insights = []

    rows = profile["shape"]["rows"]
    columns = profile["shape"]["columns"]

    insights.append(
        f"The dataset contains {rows} rows and {columns} columns."
    )

    if rows < 100:
        insights.append(
            "The dataset is quite small, so conclusions should be interpreted carefully."
        )
    elif rows > 100_000:
        insights.append(
            "The dataset is large enough for more advanced analysis or machine learning experiments."
        )

    if columns > 50:
        insights.append(
            "The dataset has many columns, so feature selection or dimensionality reduction may be useful."
        )

    return insights


def generate_missing_value_insights(profile: dict) -> list[str]:
    """
    Generate insights about missing values.
    """
    insights = []

    missing_values = profile["missing_values"]

    columns_with_missing = {
        column: values
        for column, values in missing_values.items()
        if values["missing_count"] > 0
    }

    if not columns_with_missing:
        insights.append("No missing values were detected in the dataset.")
        return insights

    insights.append(
        f"{len(columns_with_missing)} columns contain missing values."
    )

    high_missing_columns = [
        column
        for column, values in columns_with_missing.items()
        if values["missing_percentage"] >= 30
    ]

    if high_missing_columns:
        insights.append(
            "Some columns have a high percentage of missing values: "
            + ", ".join(high_missing_columns)
            + "."
        )

    moderate_missing_columns = [
        column
        for column, values in columns_with_missing.items()
        if 5 <= values["missing_percentage"] < 30
    ]

    if moderate_missing_columns:
        insights.append(
            "Some columns have a moderate amount of missing values: "
            + ", ".join(moderate_missing_columns)
            + "."
        )

    return insights


def generate_duplicate_insights(profile: dict) -> list[str]:
    """
    Generate insights about duplicated rows.
    """
    insights = []

    duplicate_rows = profile["duplicates"]["duplicate_rows"]
    duplicate_percentage = profile["duplicates"]["duplicate_percentage"]

    if duplicate_rows == 0:
        insights.append("No duplicated rows were detected.")
    else:
        insights.append(
            f"The dataset contains {duplicate_rows} duplicated rows "
            f"({duplicate_percentage}% of the dataset)."
        )

    return insights


def generate_numerical_insights(profile: dict) -> list[str]:
    """
    Generate insights about numerical columns.
    """
    insights = []

    numerical_summary = profile["numerical_summary"]

    if not numerical_summary:
        insights.append("No numerical columns were detected.")
        return insights

    insights.append(
        f"{len(numerical_summary)} numerical columns were detected."
    )

    for column, stats in numerical_summary.items():
        min_value = stats["min"]
        max_value = stats["max"]
        mean_value = stats["mean"]
        median_value = stats["median"]

        insights.append(
            f"For '{column}', values range from {min_value:.2f} to {max_value:.2f}, "
            f"with an average of {mean_value:.2f} and a median of {median_value:.2f}."
        )

        if abs(mean_value - median_value) > abs(mean_value) * 0.3 if mean_value != 0 else False:
            insights.append(
                f"The column '{column}' may be skewed because the mean and median are quite different."
            )

    return insights


def generate_categorical_insights(profile: dict) -> list[str]:
    """
    Generate insights about categorical columns.
    """
    insights = []

    categorical_summary = profile["categorical_summary"]

    if not categorical_summary:
        insights.append("No categorical columns were detected.")
        return insights

    insights.append(
        f"{len(categorical_summary)} categorical columns were detected."
    )

    for column, summary in categorical_summary.items():
        unique_values = summary["unique_values"]
        top_values = summary["top_values"]

        if not top_values:
            continue

        top_value = next(iter(top_values))
        top_count = top_values[top_value]

        insights.append(
            f"The column '{column}' contains {unique_values} unique values. "
            f"The most frequent value is '{top_value}' with {top_count} occurrences."
        )

        if unique_values > 100:
            insights.append(
                f"The column '{column}' has high cardinality, which may require special handling."
            )

    return insights


def generate_datetime_insights(profile: dict) -> list[str]:
    """
    Generate insights about datetime columns.
    """
    insights = []

    datetime_summary = profile["datetime_summary"]

    if not datetime_summary:
        insights.append("No datetime columns were detected.")
        return insights

    insights.append(
        f"{len(datetime_summary)} datetime columns were detected."
    )

    for column, summary in datetime_summary.items():
        insights.append(
            f"The datetime column '{column}' ranges from "
            f"{summary['min_date']} to {summary['max_date']}."
        )

    return insights


def generate_outlier_insights(profile: dict) -> list[str]:
    """
    Generate insights about numerical outliers.
    """
    insights = []

    outliers = profile["outliers"]

    columns_with_outliers = {
        column: values
        for column, values in outliers.items()
        if values["outlier_count"] > 0
    }

    if not columns_with_outliers:
        insights.append("No simple numerical outliers were detected using the IQR method.")
        return insights

    for column, values in columns_with_outliers.items():
        insights.append(
            f"The column '{column}' contains {values['outlier_count']} potential outliers "
            f"({values['outlier_percentage']}% of non-missing values)."
        )

    return insights


def generate_all_insights(profile: dict) -> dict:
    """
    Generate all insights from the dataset profile.
    """
    insights = {
        "shape_insights": generate_shape_insights(profile),
        "missing_value_insights": generate_missing_value_insights(profile),
        "duplicate_insights": generate_duplicate_insights(profile),
        "numerical_insights": generate_numerical_insights(profile),
        "categorical_insights": generate_categorical_insights(profile),
        "datetime_insights": generate_datetime_insights(profile),
        "outlier_insights": generate_outlier_insights(profile),
    }

    all_insights = []

    for section_insights in insights.values():
        all_insights.extend(section_insights)

    insights["all_insights"] = all_insights

    return insights


def save_insights(insights: dict, output_path: Path) -> None:
    """
    Save insights as a JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(insights, file, indent=4, ensure_ascii=False)

    print(f"\nInsights saved to: {output_path}")


def print_insights(insights: dict) -> None:
    """
    Print insights in a readable format.
    """
    print("\nGenerated insights")
    print("------------------")

    for insight in insights["all_insights"]:
        print(f"- {insight}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate simple insights from a dataset profile."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to the dataset profile JSON file. "
            "If not provided, the script lets you choose a file from reports/."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path where the insights JSON file will be saved. "
            "If not provided, it is generated from the input file name."
        ),
    )

    args = parser.parse_args()

    if args.input is None:
        input_path = select_profile_file(REPORTS_DIR)
    else:
        input_path = Path(args.input)

    if args.output is None:
        output_path = build_default_output_path(input_path)
    else:
        output_path = Path(args.output)

    print(f"\nSelected input file: {input_path}")
    print(f"Output file: {output_path}")

    profile = load_profile(input_path)
    insights = generate_all_insights(profile)

    print_insights(insights)
    save_insights(insights, output_path)


if __name__ == "__main__":
    main()