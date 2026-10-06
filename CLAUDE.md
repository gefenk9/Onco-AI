# Onco-AI

Onco-AI uses LLMs to study how oncologists choose treatment. Most of the work is on
lung cancer patients who got immunotherapy alone or with chemotherapy. Each script
reads patient cases from `cases.csv`, sends them to a model, and writes the answers
to CSV or text files. Most prompts are in Hebrew.

Authors: Gefen Keinan (gefenk9) and Assaf Morami.

## Where it runs

Real runs happen on a dedicated VM in a hospital, with access to real patient
records. API keys live in a `.env` file on the VM, never in git.

## Setup

```bash
make setup
```

This creates `.venv` and installs `requirements.txt`. By hand:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then fill in the keys
```

The code needs Python 3.10 or later (`str | None` hints).
`make clean` deletes `.venv`. The Makefile has no `run` or `convert-xlsx` targets,
though old docs mention them.

## Model access: `llm_client.py`

Every script calls `invoke_llm(system_prompt, user_prompt_text, max_tokens, temperature, provider_override=None)`.
It reads `.env` through `python-dotenv` and picks a provider from `LLM_PROVIDER`:

| Provider | Variables | Default model |
|---|---|---|
| `azure_openai` (default) | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, `OPEN_AI_MODEL` | `o1` |
| `bedrock` | AWS credentials for `boto3`, `BEDROCK_MODEL` | `eu.anthropic.claude-3-7-sonnet-20250219-v1:0`, region fixed to `eu-west-1` |
| `anthropic` | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | `claude-3-5-sonnet-20240620` |

`invoke_llm` never raises. On failure it returns a string that starts with `ERROR:`,
and callers check for that prefix.

Azure and Bedrock have tight rate limits, so scripts wait 31 seconds between calls
unless the provider is `anthropic`.

## Input

`cases.csv` in the repo root, UTF-8, with columns
`PatId, Current_Disease, Summary_Conclusions, Recommendations`.
The text in these columns is Hebrew. The copy in git holds sample cases. The real file
lives only on the VM. `.gitignore` excludes `*csv` and `*xlsx`, so new data files stay out of git.

To build `cases.csv` from a spreadsheet:

```bash
python xlsx_to_csv.py patients.xlsx cases.csv [--sheet NAME_OR_INDEX]
```

## Scripts

Run each from the repo root: `python <script>`. All paths are relative.

| Script | What it does | Writes |
|---|---|---|
| `analysis_top_20_reasons_v1.py --provider {azure_openai,bedrock,anthropic}` | Newest (Feb 2026). For each patient, asks for one of three treatments (chemo + immuno, immuno only, immuno + reduced-dose chemo) and one main reason. A second call picks the top 20 reasons across all patients and says why others were left out or merged. Checks that counts add up. Each run first renames the last run's files to `<name>.<timestamp>.csv`. | `analysis_v1_results.csv`, `uncategorized.csv`, `not_in_top_20.csv` |
| `analysis_per_case_2.py` | For each patient, the treatment type and a percentage weight for each reason in a fixed list (`ALL_REASONS`). | `analysis_per_case.csv` |
| `analysis_per_case.py` | Older version: treatment type and four main reasons per patient. | `analysis_per_case.csv` |
| `cases_to_patient_class.py` | Extracts structured fields (cancer type, age, PD-L1, ECOG status, illnesses, dose change, and so on) as JSON per patient, then prints 14 group analyses. | stdout and `cases_to_patient_class_output.txt` |
| `cases_to_cases_with_analysis.py` | Asks the model for its own treatment plan, then compares it with the doctor's and gives a similarity score from 0 to 1. | `cases_with_analysis.csv` |
| `cross_analysis.py [--max_records N]` | Sends up to N cases (default 50) in one call and asks for patterns across them, in numbers and percentages. | `cross_analysis_output.txt` |
| `cross_analysis_subjective_per_doctor.py [--max_records_per_doctor N]` | Joins `cases.csv` with `doctors.csv` on patient ID (`PatId` vs `PatID`) and asks for each doctor's habits and biases. | `raw_data/<doctor>_raw_data.csv`, stdout |
| `xlsx_to_csv.py` | Spreadsheet to CSV. | the CSV you name |
| `excel_to_csv.py` | Older converter with placeholder paths. `xlsx_to_csv.py` does the same job. | |

## Ralph

`.claude/ralph/` drives an autonomous loop: `ralph.sh` feeds `.claude/ralph/CLAUDE.md`
to `claude --print` until every story in `prd.json` passes. It built
`analysis_top_20_reasons_v1.py` from `.claude/tasks/prd-analysis-top-20-reasons-v1.md`.
`progress.txt` logs each step and the patterns learned.

## History

- Jan 2025: `main.py` picks one NCCN patient guideline and asks Claude for a treatment plan.
- May 2025: NCCN files and the guideline step removed. `main.py` became
  `cases_to_cases_with_analysis.py`, which works through `cases.csv`. Added `llm_client.py`,
  Bedrock support, the Makefile, `xlsx_to_csv.py`, and `cross_analysis.py`.
- Jun–Jul 2025: Azure OpenAI became the default. Added `cases_to_patient_class.py` and its
  14 analyses, and the per-doctor analysis.
- Sep–Oct 2025: `analysis_per_case.py` and `analysis_per_case_2.py`, with fixed reason lists.
- Feb 2026: `analysis_top_20_reasons_v1.py`, built by Ralph, merged as PR #1.

## Known problems

- `analysis_per_case.py` and `analysis_per_case_2.py` write the same output file.
- `cases_to_patient_class.py` defaults `LLM_PROVIDER` to `bedrock` when it works out the
  delay, while `llm_client.py` defaults to `azure_openai`. `BEDROCK_CLAUDE_MODEL_ID` there is unused.
- The default Anthropic model, `claude-3-5-sonnet-20240620`, is retired.
- No tests.

## Rules for Claude

- **Write by Orwell's six rules.** This covers code comments, docs, commit messages, and replies.
  1. Never use a metaphor, simile, or other figure of speech you are used to seeing in print.
  2. Never use a long word where a short one will do.
  3. If you can cut a word out, cut it out.
  4. Never use the passive where you can use the active.
  5. Never use a foreign phrase, a scientific word, or jargon if you can think of an everyday English equivalent.
  6. Break any of these rules sooner than say anything outright barbarous.

  Medical terms in prompts are exempt from rule 5 where no plain word is exact.
- **Use the ponytail skill whenever you write code.** Load it before you write, change, fix, or review code. Take the simplest change that works.
- **Every change must work on the hospital VM.** Check each change against it and say how it affects the VM:
  - Keys stay in the VM's `.env`. Do not move them, rename the variables, or change how `llm_client.py` loads them.
  - Keep script names, relative paths, input and output file names, and command-line flags, unless asked.
  - A new or changed dependency means the VM must rerun `make setup`. Say so.
  - Never widen where patient data goes: no new services, uploads, or logs beyond the model calls the scripts already make.
- **Treat patient data as sensitive.** Never commit real patient data or model output about patients. Do not paste case text into commits, issues, or chat.
- **Commit and push to `master` often.** Make small commits, one per change, and push each to `origin master`.
- **Keep the Hebrew prompts in Hebrew.** Change their meaning only when asked.
- **Leave `.claude/ralph/` alone** unless the task is about Ralph. Its CLAUDE.md gives the loop its own rules.
