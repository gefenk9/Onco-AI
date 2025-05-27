This project contains scripts for processing and analyzing oncological case data, primarily using AI/LLM capabilities via AWS Bedrock. The workflow typically involves converting patient data from an Excel file to CSV, then performing individual case analysis and cross-case analysis using LLMs.

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Scripts and Workflow](#scripts-and-workflow)
- [1. XLSX to CSV Converter (`xlsx_to_csv.py`)](#1-xlsx-to-csv-converter-xlsx_to_csvpy)
  - [Description](#description)
  - [Usage](#usage)
  - [Example:](#example)
- [2. Case Analysis with LLM (`cases_to_cases_with_analysis.py`)](#2-case-analysis-with-llm-cases_to_cases_with_analysispy)
  - [Description:](#description-1)
  - [Input CSV Format:](#input-csv-format)
  - [Usage:](#usage-1)
  - [Example Workflow Step:](#example-workflow-step)
- [3. Cross-Case Analysis (`cross_analysis.py`)](#3-cross-case-analysis-cross_analysispy)
  - [Description:](#description-2)
  - [Input CSV Format:](#input-csv-format-1)
  - [Usage:](#usage-2)
  - [Example Workflow Step:](#example-workflow-step-1)
- [Cleaning Up](#cleaning-up)

# Prerequisites

- Python 3.x
- An AWS account configured with access to AWS Bedrock and the Anthropic Claude 3.5 Sonnet model (`eu.anthropic.claude-3-5-sonnet-20240620-v1:0`). Ensure your AWS credentials (e.g., via AWS CLI configuration, IAM roles, or environment variables) are set up for `boto3` to use.
- `make` (optional, but recommended for using Makefile shortcuts).

# Setup

1.  **Clone the repo:**

    ```bash
    git clone https://github.com/gefenk9/Onco-AI
    cd Onco-AI
    ```

2.  **Create a virtual environment and install dependencies:**
    This project uses a `Makefile` to simplify setup and execution.

    ```bash
    make setup
    ```

    This command will:

    - Create a Python virtual environment in the `.venv` directory (if it doesn't exist).
    - Install all required Python packages listed in `requirements.txt`.

    Alternatively, you can set up the environment manually:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

# Scripts and Workflow

The typical workflow involves using the scripts in the following order:

# 1. XLSX to CSV Converter (`xlsx_to_csv.py`)

## Description

Converts a specified sheet from an XLSX Excel file (e.g., `patients.xlsx`) to a CSV file. This is the first step to prepare your data for the analysis scripts.

## Usage

- **Directly via Python:**

  ```bash
  # Ensure virtual environment is active: source .venv/bin/activate
  python xlsx_to_csv.py <input_excel_file.xlsx> <output_csv_file.csv> [--sheet <sheet_name_or_index>]
  ```

  - `<input_excel_file.xlsx>`: Path to your source Excel file (e.g., `patients.xlsx`).
  - `<output_csv_file.csv>`: Desired path for the generated CSV file (e.g., `patients.csv`).
  - `--sheet <sheet_name_or_index>` (optional): Specify the sheet name (e.g., `"Sheet1"`) or 0-based index (e.g., `0`). Defaults to the first sheet.

- **Via Makefile:**
  The Makefile provides the `convert-xlsx` target.

  ```bash
  # To convert 'input.xlsx' (default) to 'converted_from_xlsx.csv' (default), first sheet:
  make convert-xlsx

  # To convert a specific file, e.g., 'patients.xlsx' to 'patients.csv':
  make convert-xlsx XLSX_CONVERT_INPUT=patients.xlsx XLSX_CONVERT_OUTPUT=patients.csv

  # To specify a sheet name:
  make convert-xlsx XLSX_CONVERT_INPUT=patients.xlsx XLSX_CONVERT_OUTPUT=patients.csv XLSX_CONVERT_SHEET="Patient Data"

  # To specify a sheet by index (e.g., the second sheet):
  make convert-xlsx XLSX_CONVERT_INPUT=patients.xlsx XLSX_CONVERT_OUTPUT=patients.csv XLSX_CONVERT_SHEET=1
  ```

## Example:

To convert the first sheet of `patients.xlsx` to `patients.csv`:

```bash
make convert-xlsx XLSX_CONVERT_INPUT=patients.xlsx XLSX_CONVERT_OUTPUT=patients.csv
```

This `patients.csv` will then be used as input for the subsequent scripts (potentially after renaming to `cases.csv`).

# 2. Case Analysis with LLM (`cases_to_cases_with_analysis.py`)

## Description:

This script processes individual patient cases from a CSV file. For each case, it:

1.  Sends the patient's current disease information to an LLM (Claude 3.5 Sonnet via AWS Bedrock) to generate a suggested treatment plan.
2.  Sends the LLM's generated plan along with the original doctor's summary and recommendations to the LLM again for a comparative analysis and a similarity score.
3.  Saves the original case data, the LLM's treatment plan, the comparative analysis, and the score to a new output CSV file (default: `cases_with_analysis.csv`).

## Input CSV Format:

The script expects an input CSV file named `cases.csv` in the project root directory. This file should contain at least the following columns (headers are in Hebrew as defined in the script):

- `current_disease` (תיאור המחלה הנוכחי)
- `summary_conclusion` (סיכום ומסקנות הרופא)
- `recommendations` (המלצות הרופא)

## Usage:

- **Preparation:**
  Ensure the CSV file generated from `xlsx_to_csv.py` (e.g., `patients.csv`) is renamed or copied to `cases.csv` in the project's root directory.

  ```bash
  cp patients.csv cases.csv
  ```

- **Directly via Python:**

  ```bash
  # Ensure virtual environment is active
  python cases_to_cases_with_analysis.py
  ```

- **Via Makefile (Recommended):**
  The Makefile's `run` target is configured to execute `cases_to_cases_with_analysis.py`. If `cases_to_cases_with_analysis.py` is your `cases_to_cases_with_analysis.py` (or is called by it), use:
  ```bash
  make run
  ```
  This will process `cases.csv` and produce `cases_with_analysis.csv`.

## Example Workflow Step:

1.  Convert `patients.xlsx` to `patients.csv` (as shown in the previous section).
2.  Prepare `cases.csv`:
    ```bash
    cp patients.csv cases.csv
    ```
3.  Run the analysis:
    ```bash
    make run
    ```
    The output will be `cases_with_analysis.csv`.

# 3. Cross-Case Analysis (`cross_analysis.py`)

## Description:

This script performs a "cross-analysis" of multiple oncology cases. It reads case data (disease description and doctor's recommendations) from `cases.csv`. It then compiles this information and sends it in a single batch request to an LLM (Claude 3.5 Sonnet via AWS Bedrock). The LLM is prompted to identify common characteristics, patterns, or relationships between treatment recommendations and case features across all provided cases. The LLM's analysis is saved to a text file (default: `cross_analysis_output.txt`).

## Input CSV Format:

Similar to `cases_to_cases_with_analysis.py`, this script expects an input CSV file named `cases.csv` with at least:

- `current_disease`
- `recommendations`

## Usage:

- **Preparation:**
  Ensure `cases.csv` (e.g., from `patients.xlsx` via `xlsx_to_csv.py`) is present.

- **Directly via Python:**

  ```bash
  # Ensure virtual environment is active
  python cross_analysis.py [--max_records <number>]
  ```

  - `--max_records <number>` (optional): Limits the number of records processed from the CSV. Defaults to 50.

  Example (process all records up to the default limit):

  ```bash
  python cross_analysis.py
  ```

  Example (process up to 10 records):

  ```bash
  python cross_analysis.py --max_records 10
  ```

- **Via Makefile:**
  Currently, there isn't a dedicated Makefile target for `cross_analysis.py`. Run it directly as shown above after `make setup` and ensuring `cases.csv` is ready.

## Example Workflow Step:

1.  Convert `patients.xlsx` to `patients.csv` and then prepare `cases.csv` (as shown in previous sections).
2.  Run the cross-analysis:
    ```bash
    python cross_analysis.py
    ```
    The output will be `cross_analysis_output.txt`.

# Cleaning Up

To remove generated files (default `.csv` and `.txt` outputs) and the virtual environment directory (`.venv`):

```bash
make clean
```
