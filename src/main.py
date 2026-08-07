from __future__ import annotations

import argparse
from pathlib import Path

from data_preprocessing import (
    basic_cleaning,
    load_data as load_raw_data,
    save_data,
)

from data_profiler import (
    profile_dataset,
    save_profile,
)

from data_quality import (
    generate_quality_alerts,
    generate_quality_report,
    save_quality_report,
)

from insight_generator import (
    generate_all_insights,
    save_insights,
)

from report_generator import (
    generate_llm_report,
    generate_rule_based_report,
    save_report,
)

from process_mining import (
    print_variant_summary,
    calculate_case_durations,
    summarize_transition_durations,
    add_process_time_share,
    save_process_mining_results,
)


def build_paths(
    input_path: Path,
    output_dir: Path,
) -> dict[str, Path]:

    dataset_name = input_path.name

    if dataset_name.lower().endswith(".xes.gz"):
        dataset_name = dataset_name[:-7]
    else:
        dataset_name = input_path.stem

    return {
        "cleaned": (
            output_dir
            / "processed"
            / f"{dataset_name}_clean.csv"
        ),
        "profile": (
            output_dir
            / "reports"
            / f"{dataset_name}_profile.json"
        ),
        "quality": (
            output_dir
            / "reports"
            / f"{dataset_name}_quality.json"
        ),
        "insights": (
            output_dir
            / "reports"
            / f"{dataset_name}_insights.json"
        ),
        "report_rule": (
            output_dir
            / "reports"
            / f"{dataset_name}_rule_based_report.md"
        ),
        "report_llm": (
            output_dir
            / "reports"
            / f"{dataset_name}_llm_report.md"
        ),
        "process_mining": (
            output_dir
            / "reports"
            / "process_mining"
            / dataset_name
        ),
    }


def detect_process_columns(
    df,
    case_column: str | None = None,
    activity_column: str | None = None,
    timestamp_column: str | None = None,
) -> tuple[str, str, str]:
    """
    Automatically detect process mining columns
    when they are not provided manually.
    """

    columns = list(df.columns)

    if case_column is None:
        case_candidates = [
            "case_id",
            "case_concept_name",
            "caseid",
            "case",
            "trace_id",
            "process_id",
            "order_id",
        ]

        case_column = next(
            (
                column
                for column in case_candidates
                if column in columns
            ),
            None,
        )

    if activity_column is None:
        activity_candidates = [
            "activity_name",
            "concept_name",
            "activity",
            "event",
            "event_name",
            "status",
            "task",
            "action",
        ]

        activity_column = next(
            (
                column
                for column in activity_candidates
                if column in columns
            ),
            None,
        )

    if timestamp_column is None:
        timestamp_candidates = [
            "timestamp",
            "time_timestamp",
            "event_timestamp",
            "datetime",
            "event_time",
            "date_time",
            "date",
            "time",
        ]

        timestamp_column = next(
            (
                column
                for column in timestamp_candidates
                if column in columns
            ),
            None,
        )

    missing = []

    if case_column is None:
        missing.append("case column")

    if activity_column is None:
        missing.append("activity column")

    if timestamp_column is None:
        missing.append("timestamp column")

    if missing:
        raise ValueError(
            "Unable to automatically detect: "
            + ", ".join(missing)
            + ". Use --case-column, --activity-column "
            "and --timestamp-column manually."
        )

    print("\nProcess mining columns detected:")
    print(f"- Case: {case_column}")
    print(f"- Activity: {activity_column}")
    print(f"- Timestamp: {timestamp_column}")

    return (
        case_column,
        activity_column,
        timestamp_column,
    )


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    use_llm: bool = False,
    model: str = "mistral:latest",
    case_column: str | None = None,
    activity_column: str | None = None,
    timestamp_column: str | None = None,
) -> dict[str, Path]:

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}"
        )

    paths = build_paths(
        input_path,
        output_dir,
    )

    print("\n1/6 - Loading and cleaning data...")

    raw_df = load_raw_data(
        input_path
    )

    if raw_df.empty:
        raise ValueError(
            "The input file contains no usable rows."
        )

    cleaned_df = basic_cleaning(
        raw_df
    )

    save_data(
        cleaned_df,
        paths["cleaned"],
    )

    print("\n2/6 - Creating dataset profile...")

    profile = profile_dataset(
        cleaned_df
    )

    save_profile(
        profile,
        paths["profile"],
    )

    print("\n3/6 - Running data quality checks...")

    quality_report = generate_quality_report(
        cleaned_df
    )

    quality_report["alerts"] = (
        generate_quality_alerts(
            quality_report
        )
    )

    save_quality_report(
        quality_report,
        paths["quality"],
    )

    print("\n4/6 - Generating insights...")

    insights = generate_all_insights(
        profile
    )

    save_insights(
        insights,
        paths["insights"],
    )

    print("\n5/6 - Generating final report...")

    if use_llm:
        try:
            report = generate_llm_report(
                insights=insights,
                quality_report=quality_report,
                model=model,
            )

            report_path = paths["report_llm"]

        except (
            ConnectionError,
            TimeoutError,
            RuntimeError,
        ) as error:

            print(
                f"Ollama unavailable: {error}"
            )

            print(
                "Falling back to the rule-based report."
            )

            report = generate_rule_based_report(
                insights=insights,
                quality_report=quality_report,
            )

            report_path = paths["report_rule"]

    else:
        report = generate_rule_based_report(
            insights=insights,
            quality_report=quality_report,
        )

        report_path = paths["report_rule"]

    save_report(
        report,
        report_path,
    )

    print("\n6/6 - Running process mining...")

    try:
        (
            case_column,
            activity_column,
            timestamp_column,
        ) = detect_process_columns(
            cleaned_df,
            case_column,
            activity_column,
            timestamp_column,
        )

        print_variant_summary(
            cleaned_df,
            case_column,
            activity_column,
            timestamp_column,
        )

        case_durations = calculate_case_durations(
            cleaned_df,
            case_column,
            activity_column,
            timestamp_column,
        )

        print("\nCase Duration Overview")
        print("----------------------")

        print(
            f"Average duration: "
            f"{case_durations['duration_days'].mean():.2f} days"
        )

        print(
            f"Median duration: "
            f"{case_durations['duration_days'].median():.2f} days"
        )

        print("\nSlowest transitions")
        print("-------------------")

        transition_summary = (
            summarize_transition_durations(
                cleaned_df,
                case_column,
                activity_column,
                timestamp_column,
            )
        )

        print(
            transition_summary
            .head(10)
            .to_string(index=False)
        )

        print("\nPotential Bottlenecks")
        print("---------------------")

        bottlenecks = add_process_time_share(
            cleaned_df,
            case_column,
            activity_column,
            timestamp_column,
        )

        print(
            bottlenecks[
                [
                    "transition",
                    "case_count",
                    "frequency_ratio",
                    "average_days",
                    "median_days",
                    "process_time_share",
                    "bottleneck_score",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

        save_process_mining_results(
            cleaned_df,
            paths["process_mining"],
            case_column,
            activity_column,
            timestamp_column,
        )

    except ValueError as error:
        print("\nProcess mining skipped:")
        print(error)

    print(
        "\nPipeline completed successfully."
    )

    print(
        f"Final report: {report_path}"
    )

    return paths


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete data analysis "
            "and process mining pipeline."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the input CSV, XES or XES.GZ file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where generated files will be saved.",
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use a local Ollama model to rewrite the final report.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mistral:latest",
        help="Ollama model used when --use-llm is enabled.",
    )

    parser.add_argument(
        "--case-column",
        type=str,
        default=None,
        help=(
            "Case identifier column. "
            "Automatically detected if omitted."
        ),
    )

    parser.add_argument(
        "--activity-column",
        type=str,
        default=None,
        help=(
            "Activity column. "
            "Automatically detected if omitted."
        ),
    )

    parser.add_argument(
        "--timestamp-column",
        type=str,
        default=None,
        help=(
            "Timestamp column. "
            "Automatically detected if omitted."
        ),
    )

    args = parser.parse_args()

    run_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        use_llm=args.use_llm,
        model=args.model,
        case_column=args.case_column,
        activity_column=args.activity_column,
        timestamp_column=args.timestamp_column,
    )


if __name__ == "__main__":
    main()