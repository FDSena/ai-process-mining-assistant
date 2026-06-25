import argparse
import json
from pathlib import Path

import requests


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_MODEL = "mistral:latest"


def load_text_file(input_path: Path) -> str:
    """
    Load a text or Markdown file.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    return input_path.read_text(encoding="utf-8")


def build_report_prompt(report_content: str) -> str:
    """
    Build a prompt asking the LLM to rewrite a technical report
    in a business-friendly way.
    """
    return f"""
You are a data analyst assistant.

Rewrite the following technical dataset report into a clear, business-friendly summary.

Rules:
- Do not invent any information.
- Do not add assumptions.
- Do not say that something "may exist" if the report says it was not detected.
- If a section says empty lists like [] or empty dictionaries like {{}}, explain that no issue was detected for that point.
- Keep the explanation concise, structured, and easy to understand for a non-technical user.
- Use only the numbers and facts present in the report.

Technical report:
{report_content}
"""

def ask_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
) -> str:
    """
    Send a prompt to Ollama and return the generated response.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(
            ollama_url,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise ConnectionError(
            "Could not connect to Ollama. "
            "Make sure Ollama is running locally."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise TimeoutError(
            "Ollama took too long to respond."
        ) from exc

    data = response.json()
    return data.get("response", "").strip()


def save_llm_output(output_text: str, output_path: Path) -> None:
    """
    Save the LLM output to a Markdown file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")

    print(f"\nLLM output saved to: {output_path}")


def build_default_output_path(input_path: Path) -> Path:
    """
    Build the default output path for the LLM-generated report.
    """
    filename = input_path.stem

    if filename.endswith("_report"):
        filename = filename.replace("_report", "")

    return Path("reports") / f"{filename}_llm_report.md"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a business-friendly report using Ollama."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the Markdown report to rewrite.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path where the LLM-generated report will be saved.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="Ollama model to use. Default: mistral:latest.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if args.output is None:
        output_path = build_default_output_path(input_path)
    else:
        output_path = Path(args.output)

    print(f"\nSelected input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Model: {args.model}")

    report_content = load_text_file(input_path)
    prompt = build_report_prompt(report_content)

    llm_output = ask_ollama(prompt, model=args.model)

    print("\nLLM report preview")
    print("------------------")
    print(llm_output[:1500])

    save_llm_output(llm_output, output_path)


if __name__ == "__main__":
    main()