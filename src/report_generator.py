from pathlib import Path
import argparse
import json
from ollama_client import ask_ollama, build_report_prompt

REPORTS_DIR = Path("reports")


def select_insights_file(reports_dir: Path) -> Path:
    """
    Let the user select an insights JSON file from the reports directory.
    """
    insights_files = sorted(reports_dir.glob("*_insights.json"))

    if not insights_files:
        raise FileNotFoundError(
            f"No insights files found in {reports_dir}. "
            "Please run insight_generator.py first."
        )

    print("\nAvailable insights files in reports:\n")

    for index, file_path in enumerate(insights_files, start=1):
        print(f"[{index}] {file_path.name}")

    while True:
        choice = input("\nSelect a file number: ")

        if choice.isdigit():
            choice_index = int(choice)

            if 1 <= choice_index <= len(insights_files):
                return insights_files[choice_index - 1]

        print("Invalid choice. Please enter a valid number.")


def build_default_output_path(input_path: Path) -> Path:
    """
    Build the default Markdown report path.

    Example:
    reports/sales_data_insights.json
    becomes:
    reports/sales_data_report.md
    """
    filename = input_path.stem

    if filename.endswith("_insights"):
        filename = filename.replace("_insights", "")

    output_filename = f"{filename}_report.md"

    return REPORTS_DIR / output_filename


def load_insights(input_path: Path) -> dict:
    """
    Load the insights JSON file.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"File not found: {input_path}. "
            "Please check the path or run insight_generator.py first."
        )

    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)
    
def load_quality_report(input_path: Path | None) -> dict | None:
    """
    Load the data quality JSON report if provided.
    """
    if input_path is None:
        return None

    if not input_path.exists():
        raise FileNotFoundError(
            f"Quality report not found: {input_path}. "
            "Please run data_quality.py first or check the path."
        )

    with input_path.open("r", encoding="utf-8") as file:
        return json.load(file)	


def format_section(title: str, insights: list[str]) -> str:
    """
    Format one report section in Markdown.
    """
    lines = [f"## {title}", ""]

    if not insights:
        lines.append("No insights available for this section.")
    else:
        for insight in insights:
            lines.append(f"- {insight}")

    lines.append("")
    return "\n".join(lines)

def format_quality_section(quality_report: dict | None) -> str:
    """
    Format the data quality section in Markdown.
    """
    lines = ["## Data Quality Alerts", ""]

    if quality_report is None:
        lines.append("No data quality report was provided.")
        lines.append("")
        return "\n".join(lines)

    alerts = quality_report.get("alerts", [])

    if not alerts:
        lines.append("No data quality alerts available.")
    else:
        for alert in alerts:
            lines.append(f"- {alert}")

    lines.append("")
    lines.append("### Data Quality Details")
    lines.append("")
    lines.append(f"- Empty columns: {quality_report.get('empty_columns', [])}")
    lines.append(f"- High missing columns: {quality_report.get('high_missing_columns', {})}")
    lines.append(f"- Constant columns: {quality_report.get('constant_columns', [])}")
    lines.append(f"- High cardinality columns: {quality_report.get('high_cardinality_columns', {})}")
    lines.append(f"- Possible ID columns: {quality_report.get('possible_id_columns', {})}")
    lines.append(f"- Negative values: {quality_report.get('negative_values', {})}")
    lines.append(f"- Date columns: {quality_report.get('date_columns', [])}")
    lines.append(f"- Invalid dates: {quality_report.get('invalid_dates', {})}")
    lines.append("")

    return "\n".join(lines)

def generate_rule_based_report(insights: dict, quality_report: dict | None = None) -> str:
    """
    Generate a readable Markdown report without using an LLM.
    """
    report_parts = [
        "# Dataset Analysis Report",
        "",
        "This report was generated automatically from the dataset profile and rule-based insights.",
        "It summarizes the structure, data quality, column types, and potential points of attention in the dataset.",
        "",
    ]
    
    report_parts.append(format_quality_section(quality_report))
    
    sections = [
        ("Dataset Overview", insights.get("shape_insights", [])),
        ("Missing Values", insights.get("missing_value_insights", [])),
        ("Duplicated Rows", insights.get("duplicate_insights", [])),
        ("Numerical Columns", insights.get("numerical_insights", [])),
        ("Categorical Columns", insights.get("categorical_insights", [])),
        ("Datetime Columns", insights.get("datetime_insights", [])),
        ("Outliers", insights.get("outlier_insights", [])),
    ]

    for title, section_insights in sections:
        report_parts.append(format_section(title, section_insights))

    report_parts.append("## Next Steps")
    report_parts.append("")
    report_parts.append("- Review the columns with high cardinality before using them in machine learning models.")
    report_parts.append("- Check whether the dataset contains a meaningful target variable for prediction.")
    report_parts.append("- Create visualizations for the most important numerical and categorical columns.")
    report_parts.append("- Use an LLM layer later to rewrite this report in a more business-friendly style.")
    report_parts.append("")

    return "\n".join(report_parts)


def generate_llm_report(
    insights: dict,
    quality_report: dict | None = None,
    model: str = "mistral:latest",
) -> str:
    """
    Generate a business-friendly report using Ollama.
    """
    rule_based_report = generate_rule_based_report(insights, quality_report)
    prompt = build_report_prompt(rule_based_report)

    return ask_ollama(prompt, model=model)


def save_report(report: str, output_path: Path) -> None:
    """
    Save the Markdown report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        file.write(report)

    print(f"\nReport saved to: {output_path}")


def print_report_preview(report: str, max_lines: int = 30) -> None:
    """
    Print the first lines of the generated report.
    """
    print("\nReport preview")
    print("--------------")

    lines = report.splitlines()

    for line in lines[:max_lines]:
        print(line)

    if len(lines) > max_lines:
        print("\n... report preview truncated ...")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown report from dataset insights."
    )

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help=(
            "Path to the insights JSON file. "
            "If not provided, the script lets you choose a file from reports/."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Path where the Markdown report will be saved. "
            "If not provided, it is generated from the input file name."
        ),
    )
    
    parser.add_argument(
        "--quality",
        type=str,
        default=None,
        help=(
            "Optional path to the data quality JSON report. "
            "If provided, data quality alerts will be included in the report."
        ),
    )

    parser.add_argument(
        "--use-llm",
        action="store_true",
        help=(
            "Use LLM-powered report generation. "
            "Currently this is a placeholder and falls back to rule-based generation."
        ),
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="mistral:latest",
        help="Ollama model to use when --use-llm is enabled.",
    )

    args = parser.parse_args()

    if args.input is None:
        input_path = select_insights_file(REPORTS_DIR)
    else:
        input_path = Path(args.input)

    if args.output is None:
        output_path = build_default_output_path(input_path)
    else:
        output_path = Path(args.output)

    print(f"\nSelected input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"LLM mode: {'on' if args.use_llm else 'off'}")

    insights = load_insights(input_path)
    
    quality_path = Path(args.quality) if args.quality is not None else None
    quality_report = load_quality_report(quality_path)
    
    if args.use_llm:
        report = generate_llm_report(
            insights,
            quality_report,
            model=args.model,
        )
    else:
        report = generate_rule_based_report(insights, quality_report)


    print_report_preview(report)
    save_report(report, output_path)


if __name__ == "__main__":
    main()