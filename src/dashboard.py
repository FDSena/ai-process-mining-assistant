from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="AI Process Mining Assistant",
    page_icon="📊",
    layout="wide",
)


OUTPUT_DIR = Path("output/reports/process_mining")


# -------------------------------------------------
# HELPERS
# -------------------------------------------------

def get_available_datasets() -> list[str]:
    if not OUTPUT_DIR.exists():
        return []

    return sorted(
        folder.name
        for folder in OUTPUT_DIR.iterdir()
        if folder.is_dir()
    )


def load_results(
    dataset_name: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    dataset_dir = OUTPUT_DIR / dataset_name

    variants_path = dataset_dir / "process_variants.csv"
    durations_path = dataset_dir / "case_durations.csv"
    bottlenecks_path = dataset_dir / "bottlenecks.csv"

    required_files = [
        variants_path,
        durations_path,
        bottlenecks_path,
    ]

    missing_files = [
        path.name
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing process mining files: "
            + ", ".join(missing_files)
        )

    variants = pd.read_csv(variants_path)
    durations = pd.read_csv(durations_path)
    bottlenecks = pd.read_csv(bottlenecks_path)

    return variants, durations, bottlenecks


def shorten_text(
    text: str,
    max_length: int = 75,
) -> str:
    text = str(text)

    if len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..."


def format_transition(text: str) -> str:
    return shorten_text(
        text,
        max_length=65,
    )


# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("AI Process Mining Assistant")

st.caption(
    "Analyze process variants, durations and bottlenecks "
    "from event logs."
)


datasets = get_available_datasets()

if not datasets:
    st.warning(
        "No process mining results found. "
        "Run the analysis pipeline first."
    )
    st.stop()


selected_dataset = st.selectbox(
    "Dataset",
    datasets,
)


try:
    variants, durations, bottlenecks = load_results(
        selected_dataset
    )

except FileNotFoundError as error:
    st.error(str(error))
    st.stop()


# -------------------------------------------------
# METRICS
# -------------------------------------------------

total_cases = len(durations)

number_of_variants = len(variants)

average_duration = (
    durations["duration_days"].mean()
)

median_duration = (
    durations["duration_days"].median()
)


top_bottleneck = (
    bottlenecks
    .sort_values(
        "bottleneck_score",
        ascending=False,
    )
    .iloc[0]
)


st.divider()

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Cases",
    f"{total_cases:,}",
)

col2.metric(
    "Process variants",
    f"{number_of_variants:,}",
)

col3.metric(
    "Average duration",
    f"{average_duration:.2f} days",
)

col4.metric(
    "Median duration",
    f"{median_duration:.2f} days",
)


st.info(
    "Main bottleneck: "
    f"{top_bottleneck['transition']} "
    f"— {top_bottleneck['average_days']:.2f} days average delay "
    f"across {int(top_bottleneck['case_count']):,} transitions."
)


# -------------------------------------------------
# TABS
# -------------------------------------------------

tab_overview, tab_variants, tab_bottlenecks, tab_durations = (
    st.tabs(
        [
            "Overview",
            "Variants",
            "Bottlenecks",
            "Durations",
        ]
    )
)


# =================================================
# OVERVIEW
# =================================================

with tab_overview:

    st.header("Process Overview")

    left, right = st.columns(2)

    # ---------------------------------------------
    # TOP VARIANTS
    # ---------------------------------------------

    with left:

        st.subheader("Top 5 Process Variants")

        top_variants = (
            variants
            .sort_values(
                "case_count",
                ascending=False,
            )
            .head(5)
            .copy()
        )

        top_variants["variant_number"] = [
            f"Variant #{i}"
            for i in range(
                1,
                len(top_variants) + 1,
            )
        ]

        chart_variants = (
            top_variants
            .sort_values(
                "case_count",
                ascending=True,
            )
        )

        fig_variants = px.bar(
            chart_variants,
            x="case_count",
            y="variant_number",
            orientation="h",
            text="case_count",
            labels={
                "case_count": "Cases",
                "variant_number": "",
            },
            hover_data={
                "variant": True,
                "percentage": ":.2f",
                "case_count": True,
                "variant_number": False,
            },
        )

        fig_variants.update_traces(
            textposition="outside",
        )

        fig_variants.update_layout(
            height=420,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
            yaxis_title=None,
            xaxis_title="Number of cases",
        )

        st.plotly_chart(
            fig_variants,
            use_container_width=True,
        )

    # ---------------------------------------------
    # TOP BOTTLENECKS
    # ---------------------------------------------

    with right:

        st.subheader("Top 5 Bottlenecks")

        top_bottlenecks = (
            bottlenecks
            .sort_values(
                "bottleneck_score",
                ascending=False,
            )
            .head(5)
            .copy()
        )

        top_bottlenecks["short_transition"] = (
            top_bottlenecks["transition"]
            .apply(format_transition)
        )

        chart_bottlenecks = (
            top_bottlenecks
            .sort_values(
                "bottleneck_score",
                ascending=True,
            )
        )

        fig_bottlenecks = px.bar(
            chart_bottlenecks,
            x="bottleneck_score",
            y="short_transition",
            orientation="h",
            text="bottleneck_score",
            labels={
                "bottleneck_score": "Bottleneck score",
                "short_transition": "",
            },
            hover_data={
                "transition": True,
                "case_count": True,
                "average_days": ":.2f",
                "process_time_share": ":.2f",
                "short_transition": False,
            },
        )

        fig_bottlenecks.update_traces(
            texttemplate="%{x:.2f}",
            textposition="outside",
        )

        fig_bottlenecks.update_layout(
            height=420,
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
            yaxis_title=None,
            xaxis_title="Bottleneck score",
        )

        st.plotly_chart(
            fig_bottlenecks,
            use_container_width=True,
        )


    st.subheader("Main Findings")

    main_variant = (
        variants
        .sort_values(
            "case_count",
            ascending=False,
        )
        .iloc[0]
    )

    col_a, col_b, col_c = st.columns(3)

    col_a.metric(
        "Most common variant",
        f"{main_variant['percentage']:.2f}%",
    )

    col_b.metric(
        "Main bottleneck delay",
        f"{top_bottleneck['average_days']:.2f} days",
    )

    col_c.metric(
        "Main bottleneck time share",
        f"{top_bottleneck['process_time_share']:.2f}%",
    )


# =================================================
# VARIANTS
# =================================================

with tab_variants:

    st.header("Process Variants")

    st.write(
        "A process variant represents one complete "
        "sequence of activities followed by a case."
    )


    top_variants = (
        variants
        .sort_values(
            "case_count",
            ascending=False,
        )
        .head(10)
        .copy()
    )

    top_variants["variant_number"] = [
        f"Variant #{i}"
        for i in range(
            1,
            len(top_variants) + 1,
        )
    ]


    chart_variants = (
        top_variants
        .sort_values(
            "case_count",
            ascending=True,
        )
    )


    fig_variants_full = px.bar(
        chart_variants,
        x="case_count",
        y="variant_number",
        orientation="h",
        text="case_count",
        labels={
            "case_count": "Number of cases",
            "variant_number": "",
        },
        hover_data={
            "variant": True,
            "percentage": ":.2f",
            "case_count": True,
            "variant_number": False,
        },
    )


    fig_variants_full.update_traces(
        textposition="outside",
    )


    fig_variants_full.update_layout(
        height=520,
        yaxis_title=None,
        xaxis_title="Number of cases",
    )


    st.plotly_chart(
        fig_variants_full,
        use_container_width=True,
    )


    st.subheader("Variant Details")


    for _, row in top_variants.iterrows():

        with st.expander(
            f"{row['variant_number']} "
            f"— {int(row['case_count']):,} cases "
            f"({row['percentage']:.2f}%)"
        ):

            st.write(
                row["variant"]
            )


# =================================================
# BOTTLENECKS
# =================================================

with tab_bottlenecks:

    st.header("Main Process Bottlenecks")

    st.write(
        "The bottleneck score combines transition duration "
        "and frequency. A high score indicates a delay that "
        "affects a meaningful part of the process."
    )


    top_bottlenecks = (
        bottlenecks
        .sort_values(
            "bottleneck_score",
            ascending=False,
        )
        .head(10)
        .copy()
    )


    top_bottlenecks["short_transition"] = (
        top_bottlenecks["transition"]
        .apply(format_transition)
    )


    chart_bottlenecks = (
        top_bottlenecks
        .sort_values(
            "bottleneck_score",
            ascending=True,
        )
    )


    fig_bottlenecks_full = px.bar(
        chart_bottlenecks,
        x="bottleneck_score",
        y="short_transition",
        orientation="h",
        text="bottleneck_score",
        labels={
            "bottleneck_score": "Bottleneck score",
            "short_transition": "",
        },
        hover_data={
            "transition": True,
            "case_count": True,
            "average_days": ":.2f",
            "median_days": ":.2f",
            "process_time_share": ":.2f",
            "short_transition": False,
        },
    )


    fig_bottlenecks_full.update_traces(
        texttemplate="%{x:.2f}",
        textposition="outside",
    )


    fig_bottlenecks_full.update_layout(
        height=600,
        yaxis_title=None,
        xaxis_title="Bottleneck score",
    )


    st.plotly_chart(
        fig_bottlenecks_full,
        use_container_width=True,
    )


    st.subheader("Bottleneck Details")


    bottleneck_table = (
        top_bottlenecks[
            [
                "transition",
                "case_count",
                "average_days",
                "median_days",
                "process_time_share",
                "bottleneck_score",
            ]
        ]
        .rename(
            columns={
                "transition": "Transition",
                "case_count": "Cases",
                "average_days": "Average delay (days)",
                "median_days": "Median delay (days)",
                "process_time_share": "Process time share (%)",
                "bottleneck_score": "Score",
            }
        )
    )


    st.dataframe(
        bottleneck_table,
        use_container_width=True,
        hide_index=True,
    )


# =================================================
# DURATIONS
# =================================================

with tab_durations:

    st.header("Case Duration")

    st.write(
        "Case duration is measured between the first "
        "and last recorded event of each process case."
    )


    duration_bins = [
        0,
        10,
        20,
        30,
        60,
        90,
        120,
        180,
        365,
        float("inf"),
    ]


    duration_labels = [
        "0–10 days",
        "10–20 days",
        "20–30 days",
        "30–60 days",
        "60–90 days",
        "90–120 days",
        "120–180 days",
        "180–365 days",
        "365+ days",
    ]


    duration_categories = pd.cut(
        durations["duration_days"],
        bins=duration_bins,
        labels=duration_labels,
        right=False,
    )


    duration_distribution = (
        duration_categories
        .value_counts()
        .reindex(duration_labels)
        .fillna(0)
        .astype(int)
        .reset_index()
    )


    duration_distribution.columns = [
        "Duration range",
        "Cases",
    ]


    fig_duration = px.bar(
        duration_distribution,
        x="Duration range",
        y="Cases",
        text="Cases",
        labels={
            "Duration range": "Case duration",
            "Cases": "Number of cases",
        },
    )


    fig_duration.update_traces(
        textposition="outside",
    )


    fig_duration.update_layout(
        height=450,
        xaxis_title="Case duration",
        yaxis_title="Number of cases",
    )


    st.plotly_chart(
        fig_duration,
        use_container_width=True,
    )


    # ---------------------------------------------
    # SLOWEST TRANSITIONS
    # ---------------------------------------------

    st.subheader(
        "Slowest Relevant Transitions"
    )

    st.write(
        "Transitions with fewer than 10 occurrences "
        "are excluded to avoid highlighting isolated cases."
    )


    relevant_transitions = (
        bottlenecks[
            bottlenecks["case_count"] >= 10
        ]
        .sort_values(
            "average_days",
            ascending=False,
        )
        .head(10)
        .copy()
    )


    relevant_transitions["short_transition"] = (
        relevant_transitions["transition"]
        .apply(format_transition)
    )


    relevant_chart = (
        relevant_transitions
        .sort_values(
            "average_days",
            ascending=True,
        )
    )


    fig_slowest = px.bar(
        relevant_chart,
        x="average_days",
        y="short_transition",
        orientation="h",
        text="average_days",
        labels={
            "average_days": "Average delay (days)",
            "short_transition": "",
        },
        hover_data={
            "transition": True,
            "case_count": True,
            "median_days": ":.2f",
            "min_days": ":.2f",
            "max_days": ":.2f",
            "short_transition": False,
        },
    )


    fig_slowest.update_traces(
        texttemplate="%{x:.1f} days",
        textposition="outside",
    )


    fig_slowest.update_layout(
        height=550,
        yaxis_title=None,
        xaxis_title="Average delay (days)",
    )


    st.plotly_chart(
        fig_slowest,
        use_container_width=True,
    )


    st.subheader(
        "Transition Details"
    )


    slowest_table = (
        relevant_transitions[
            [
                "transition",
                "case_count",
                "average_days",
                "median_days",
                "min_days",
                "max_days",
            ]
        ]
        .rename(
            columns={
                "transition": "Transition",
                "case_count": "Cases",
                "average_days": "Average delay (days)",
                "median_days": "Median delay (days)",
                "min_days": "Minimum (days)",
                "max_days": "Maximum (days)",
            }
        )
    )


    st.dataframe(
        slowest_table,
        use_container_width=True,
        hide_index=True,
    )