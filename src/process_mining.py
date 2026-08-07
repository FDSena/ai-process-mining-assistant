import pandas as pd
from pathlib import Path


def validate_event_log(
    df: pd.DataFrame,
    case_column: str,
    activity_column: str,
    timestamp_column: str,
) -> None:
    """
    Check that the dataset contains the required
    columns for process mining.
    """
    required_columns = {
        case_column,
        activity_column,
        timestamp_column,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns for process mining: "
            + ", ".join(sorted(missing_columns))
        )


def prepare_event_log(
    df: pd.DataFrame,
    case_column: str,
    activity_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """
    Prepare the event log for process mining.
    """
    validate_event_log(
        df,
        case_column,
        activity_column,
        timestamp_column,
    )

    prepared_df = df.copy()

    prepared_df[timestamp_column] = pd.to_datetime(
        prepared_df[timestamp_column],
        errors="coerce",
    )

    invalid_timestamps = (
        prepared_df[timestamp_column]
        .isna()
        .sum()
    )

    if invalid_timestamps > 0:
        raise ValueError(
            f"{invalid_timestamps} invalid timestamps detected."
        )

    prepared_df = prepared_df.sort_values(
        [
            case_column,
            timestamp_column,
        ]
    ).reset_index(drop=True)

    return prepared_df


def build_case_variants(
    df: pd.DataFrame,
    case_column: str,
    activity_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """
    Reconstruct the ordered activity sequence
    for each case.
    """
    prepared_df = prepare_event_log(
        df,
        case_column,
        activity_column,
        timestamp_column,
    )

    case_variants = (
        prepared_df
        .groupby(case_column)[activity_column]
        .agg(tuple)
        .reset_index(name="variant")
    )

    return case_variants


def summarize_variants(
    df: pd.DataFrame,
    case_column: str,
    activity_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """
    Count how many cases follow each process variant.
    """
    case_variants = build_case_variants(
        df,
        case_column,
        activity_column,
        timestamp_column,
    )

    total_cases = len(case_variants)

    if total_cases == 0:
        raise ValueError(
            "No valid process cases were detected."
        )

    summary = (
        case_variants["variant"]
        .value_counts()
        .reset_index()
    )

    summary.columns = [
        "variant",
        "case_count",
    ]

    summary["percentage"] = (
        summary["case_count"]
        / total_cases
        * 100
    ).round(4)

    return summary


def print_variant_summary(
    df: pd.DataFrame,
    case_column: str,
    activity_column: str,
    timestamp_column: str,
) -> None:
    """
    Print a basic overview of process variants.
    """
    variants = summarize_variants(
        df,
        case_column,
        activity_column,
        timestamp_column,
    )

    total_cases = df[case_column].nunique()

    print("\nProcess Mining Overview")
    print("-----------------------")

    print(f"Total cases: {total_cases}")
    print(f"Number of variants: {len(variants)}")

    print("\nTop process variants:")

    for index, row in variants.head(10).iterrows():
        path = " -> ".join(
            str(activity)
            for activity in row["variant"]
        )

        print(
            f"\nVariant {index + 1}"
            f"\nCases: {row['case_count']}"
            f" ({row['percentage']:.2f}%)"
            f"\nPath: {path}"
        )


def calculate_case_durations(
    df: pd.DataFrame,
    case_column: str,
    activity_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """
    Calculate total duration for each case.
    """
    prepared_df = prepare_event_log(
        df,
        case_column,
        activity_column,
        timestamp_column,
    )

    case_durations = (
        prepared_df
        .groupby(case_column)[timestamp_column]
        .agg(["min", "max"])
        .reset_index()
    )

    case_durations["duration"] = (
        case_durations["max"]
        - case_durations["min"]
    )

    case_durations["duration_days"] = (
        case_durations["duration"]
        .dt.total_seconds()
        / 86400
    )

    return case_durations


def calculate_transition_durations(
    df: pd.DataFrame,
    case_column: str,
    activity_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """
    Calculate the time between consecutive activities.
    """
    prepared_df = prepare_event_log(
        df,
        case_column,
        activity_column,
        timestamp_column,
    )

    prepared_df["next_activity"] = (
        prepared_df
        .groupby(case_column)[activity_column]
        .shift(-1)
    )

    prepared_df["next_timestamp"] = (
        prepared_df
        .groupby(case_column)[timestamp_column]
        .shift(-1)
    )

    prepared_df["transition_duration"] = (
        prepared_df["next_timestamp"]
        - prepared_df[timestamp_column]
    )

    transitions = prepared_df.dropna(
        subset=["next_activity"]
    ).copy()

    transitions["transition"] = (
        transitions[activity_column].astype(str)
        + " -> "
        + transitions["next_activity"].astype(str)
    )

    transitions["duration_days"] = (
        transitions["transition_duration"]
        .dt.total_seconds()
        / 86400
    )

    return transitions


def summarize_transition_durations(
    df: pd.DataFrame,
    case_column: str,
    activity_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """
    Summarize duration statistics for each transition.
    """
    transitions = calculate_transition_durations(
        df,
        case_column,
        activity_column,
        timestamp_column,
    )

    summary = (
        transitions
        .groupby("transition")["duration_days"]
        .agg(
            case_count="count",
            average_days="mean",
            median_days="median",
            min_days="min",
            max_days="max",
        )
        .reset_index()
    )

    summary = summary.sort_values(
        "average_days",
        ascending=False,
    )

    for column in [
        "average_days",
        "median_days",
        "min_days",
        "max_days",
    ]:
        summary[column] = (
            summary[column].round(2)
        )

    return summary


def calculate_bottleneck_scores(
    df: pd.DataFrame,
    case_column: str,
    activity_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """
    Rank transitions using frequency and duration.
    """
    transition_summary = (
        summarize_transition_durations(
            df,
            case_column,
            activity_column,
            timestamp_column,
        )
    )

    total_transitions = (
        transition_summary["case_count"].sum()
    )

    if total_transitions == 0:
        raise ValueError(
            "No process transitions were detected."
        )

    transition_summary["frequency_ratio"] = (
        transition_summary["case_count"]
        / total_transitions
    )

    transition_summary["bottleneck_score"] = (
        transition_summary["average_days"]
        * transition_summary["frequency_ratio"]
    )

    transition_summary["frequency_ratio"] = (
        transition_summary["frequency_ratio"]
        * 100
    ).round(2)

    transition_summary["bottleneck_score"] = (
        transition_summary["bottleneck_score"]
        .round(4)
    )

    transition_summary = (
        transition_summary.sort_values(
            "bottleneck_score",
            ascending=False,
        )
    )

    return transition_summary


def add_process_time_share(
    df: pd.DataFrame,
    case_column: str,
    activity_column: str,
    timestamp_column: str,
) -> pd.DataFrame:
    """
    Estimate how much each transition contributes
    to total process duration.
    """
    bottlenecks = calculate_bottleneck_scores(
        df,
        case_column,
        activity_column,
        timestamp_column,
    )

    case_durations = calculate_case_durations(
        df,
        case_column,
        activity_column,
        timestamp_column,
    )

    average_case_duration = (
        case_durations["duration_days"].mean()
    )

    total_cases = df[case_column].nunique()

    if total_cases == 0:
        raise ValueError(
            "No process cases were detected."
        )

    if average_case_duration == 0:
        raise ValueError(
            "Average process duration is zero."
        )

    bottlenecks["process_time_share"] = (
        bottlenecks["average_days"]
        * bottlenecks["case_count"]
        / (
            average_case_duration
            * total_cases
        )
        * 100
    ).round(2)

    return bottlenecks


def save_process_mining_results(
    df: pd.DataFrame,
    output_dir: Path,
    case_column: str,
    activity_column: str,
    timestamp_column: str,
) -> None:
    """
    Save process mining results as CSV files.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    variants = summarize_variants(
        df,
        case_column,
        activity_column,
        timestamp_column,
    )

    variants = variants.copy()

    variants["variant"] = (
        variants["variant"].apply(
            lambda activities: " -> ".join(
                str(activity)
                for activity in activities
            )
        )
    )

    variants.to_csv(
        output_dir / "process_variants.csv",
        index=False,
    )

    case_durations = calculate_case_durations(
        df,
        case_column,
        activity_column,
        timestamp_column,
    )

    case_durations.to_csv(
        output_dir / "case_durations.csv",
        index=False,
    )

    bottlenecks = add_process_time_share(
        df,
        case_column,
        activity_column,
        timestamp_column,
    )

    bottlenecks.to_csv(
        output_dir / "bottlenecks.csv",
        index=False,
    )

    print(
        f"\nProcess mining results saved to: "
        f"{output_dir}"
    )