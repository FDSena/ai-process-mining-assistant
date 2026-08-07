# AI Process Mining Assistant

AI Process Mining Assistant is a Python application for automated dataset analysis and process mining.

The tool can load CSV and XES event logs, clean and profile the data, detect data quality issues, generate rule-based insights, optionally rewrite reports with a local Ollama model, and analyze business processes through variants, durations, transitions, and bottlenecks.

An interactive Streamlit dashboard makes the main process-mining results easier to explore.

## Why this project

Many data projects begin with the same problem: receiving an unfamiliar dataset and needing to understand its structure, quality, and operational meaning before building a model.

This project combines two complementary workflows:

1. **Generic dataset analysis** for understanding and validating tabular data.
2. **Process mining** for event logs containing a case identifier, an activity, and a timestamp.

The goal is to build a reusable end-to-end tool rather than a script designed for a single dataset.

## Features

### Data ingestion
- Supports CSV files.
- Supports XES and compressed XES event logs.
- Handles several CSV encodings and separators.
- Converts event logs into pandas DataFrames.

### Data preprocessing
- Cleans column names.
- Removes duplicated rows.
- Saves a cleaned CSV version of the dataset.

### Dataset profiling
- Dataset shape.
- Numerical, categorical, datetime, boolean, and identifier columns.
- Missing values.
- Numerical statistics.
- Categorical distributions.
- Datetime ranges.
- Duplicate detection.
- Basic IQR outlier detection.

### Data quality checks
- Empty columns.
- High missing-value ratios.
- Constant columns.
- High-cardinality columns.
- Possible identifier columns.
- Negative numerical values.
- Invalid date values.

### Insight and report generation
- Rule-based insights generated from the profile.
- Markdown report generation.
- Optional local LLM report rewriting with Ollama.
- Automatic fallback to the rule-based report if the LLM output fails validation.

### Process mining
When the dataset is an event log, the application can:
- Automatically detect likely case, activity, and timestamp columns.
- Allow manual column overrides from the CLI.
- Reconstruct complete process variants.
- Count variant frequency and percentage.
- Calculate case duration.
- Calculate transition duration.
- Rank slow transitions.
- Estimate process-time contribution.
- Compute a bottleneck score based on transition duration and frequency.
- Save process-mining results as CSV files.

### Dashboard
The Streamlit dashboard displays:
- Total number of cases.
- Number of process variants.
- Average and median case duration.
- Most common process variants.
- Main process bottlenecks.
- Case-duration distribution.
- Slowest relevant transitions.
- Interactive Plotly charts and tables.

## Supported input formats

```text
.csv
.xes
.xes.gz
```

For process mining, the event log needs three logical fields:

```text
Case identifier
Activity
Timestamp
```

The application tries to detect these automatically.

Examples of supported naming conventions include:

```text
case_id / activity_name / timestamp
case_id / concept_name / time_timestamp
order_id / status / event_time
```

If automatic detection is not correct, the columns can be specified manually.

## Example usage

### CSV
```bash
py src/main.py --input data/raw/Insurance_claims_event_log.csv
```

### XES
```bash
py src/main.py --input data/raw/PermitLog.xes
```

### Local LLM report with Ollama
```bash
py src/main.py --input data/raw/Insurance_claims_event_log.csv --use-llm
```

### Manual process-column configuration
```bash
py src/main.py --input data/raw/orders.csv --case-column order_id --activity-column status --timestamp-column event_time
```

## Pipeline

```text
CSV / XES event log
        ↓
Data loading
        ↓
Preprocessing
        ↓
Dataset profiling
        ↓
Data quality checks
        ↓
Rule-based insights
        ↓
Markdown report
        ↓
Optional Ollama rewriting
        ↓
Process mining
        ↓
Variants / durations / transitions / bottlenecks
        ↓
Streamlit + Plotly dashboard
```

## Dashboard

Launch the dashboard with:

```bash
py -m streamlit run src/dashboard.py
```

The dashboard reads the process-mining results generated under:

```text
output/reports/process_mining/
```

## Project structure

```text
ai-process-mining-assistant/
├── data/
│   └── raw/
├── output/
│   ├── processed/
│   └── reports/
│       └── process_mining/
├── src/
│   ├── main.py
│   ├── data_preprocessing.py
│   ├── data_profiler.py
│   ├── data_quality.py
│   ├── insight_generator.py
│   ├── report_generator.py
│   ├── ollama_client.py
│   ├── process_mining.py
│   └── dashboard.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Stack
- Python
- pandas
- PM4Py
- Streamlit
- Plotly
- Ollama

## Installation

```bash
py -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
py -m pip install -r requirements.txt
```

## Ollama

The LLM layer is optional.

To use it, Ollama must be installed and running locally with a compatible model, for example:

```bash
ollama pull mistral
```

Then run:

```bash
py src/main.py --input data/raw/your_file.csv --use-llm
```

If Ollama is unavailable or produces output that fails validation, the application automatically falls back to the deterministic rule-based report.

## Current status

Core V1 completed.

- [x] CSV loading
- [x] XES / XES.GZ loading
- [x] Data preprocessing
- [x] Generic dataset profiling
- [x] Data quality checks
- [x] Automatic insight generation
- [x] Rule-based Markdown report
- [x] Local Ollama integration
- [x] LLM fallback and basic output validation
- [x] Process-column auto-detection
- [x] Process variant discovery
- [x] Case-duration analysis
- [x] Transition-duration analysis
- [x] Bottleneck scoring
- [x] Process-mining CSV exports
- [x] Streamlit dashboard
- [x] Plotly visualizations
- [x] Tested on CSV and XES event logs

## Possible future improvements
- Automated tests with pytest.
- CI with GitHub Actions.
- Docker support.
- Direct upload and analysis from the Streamlit interface.
- Process graph / directly-follows graph visualization.
- More advanced conformance checking with PM4Py.
- Predictive models for delayed cases.
- Additional LLM validation and structured output.
