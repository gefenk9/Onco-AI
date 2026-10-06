# Onco-AI

Onco-AI asks Claude for a cancer treatment plan, in Hebrew, for one patient case.
It grounds the answer in one NCCN patient guideline that Claude picks first.
Authors: gefenk9 and Assaf Morami.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with `source .venv/bin/activate`.
`claude-env/` is an old venv. Ignore it.

Run every script from the repo root. All paths are relative.

## How it works

`main.py` makes three calls to the Anthropic API:

1. **Pick a guideline.** The system prompt lists every file in
   `guidelines_descriptions.json` with its 50-word summary. Claude replies with one
   filename. If that file does not exist, the script falls back to `all-patient.txt`.
2. **Plan with the guideline ("LLM+RAG").** The script loads the whole guideline,
   collapses whitespace, and appends it to a Hebrew system prompt (`SYSTEM_PROMPT_BASE`).
   That prompt asks for treatment options, blood tests, side-effect care, and patient
   education, each with a reason, in Hebrew only.
3. **Plan without the guideline ("LLM").** The same Hebrew prompt, no guideline. This
   is the baseline to compare against step 2.

Input is `user_prompt.txt`: one free-text patient case in Hebrew. Output goes to stdout,
with token usage for each call.

## Files

| Path | Role |
|---|---|
| `main.py` | The pipeline above. |
| `main_anthropic_gefen.py` | Untracked copy of `main.py` with an empty key. |
| `user_prompt.txt` | The patient case to run. |
| `NCCN_Guidlines/` | 72 NCCN patient guidelines as plain text, from PDFs. The typo in the name is load-bearing; code refers to it. |
| `guidelines_descriptions.json` | Filename → 50-word disease summary. Built by `txt_to_description.py`. |
| `txt_to_description.py` | Summarises the first 300 lines of each guideline. Rerun it after you add or change a guideline. |
| `excel_to_csv.py` | Exports chosen columns (`Summary_conclusion`, `Current_disease`) from a spreadsheet to CSV. Input and output paths are placeholders. |
| `Patients.xlsx` | Patient cases. |

## History

- `b8fcb80` first: guidelines, a single call with all guidelines in the prompt.
- `e4d892d` rag: two steps, pick a file from 300-line previews, then answer with it.
- `76474f6`: previews replaced by pre-built summaries in JSON, to cut tokens.
- `2199392`: prompt asks for specific blood tests.
- `3cfb722`: `excel_to_csv.py`.
- Uncommitted: the baseline call without a guideline.

## Known problems

- Model `claude-3-5-sonnet-20241022` is retired, and `anthropic==0.42.0` is old.
- The fallback guideline, `all-patient.txt`, covers acute lymphoblastic leukemia. It is a poor default for most cases.
- No tests.

## Rules for Claude

- **Write by Orwell's six rules.** This covers code comments, docs, commit messages, and replies.
  1. Never use a metaphor, simile, or other figure of speech you are used to seeing in print.
  2. Never use a long word where a short one will do.
  3. If you can cut a word out, cut it out.
  4. Never use the passive where you can use the active.
  5. Never use a foreign phrase, a scientific word, or jargon if you can think of an everyday English equivalent.
  6. Break any of these rules sooner than say anything outright barbarous.

  Medical terms in prompts and guidelines are exempt from rule 5 where no plain word is exact.
- **Use the ponytail skill whenever you write code.** Load it before you write, change, fix, or review code. Take the simplest change that works.
- **Every change must work on the hospital VM.** The project runs on a dedicated VM in a hospital, with access to real patient records. Check each change against it and say how it affects the VM:
  - The VM depends on the API key hard-coded in the scripts. Keep it there; do not move it to an environment variable or config file. The repo is private for this reason.
  - Keep script names, relative paths, the `NCCN_Guidlines/` name, and how scripts are run, unless asked.
  - A new or changed dependency means the VM must rerun `pip install -r requirements.txt`. Say so.
  - Never widen where patient data goes: no new services, uploads, logs, or files outside the repo.
- **Commit and push to `master` often.** Make small commits, one per change, and push each to `origin master`. Never commit `user_prompt.txt` or `Patients.xlsx` changes unless asked.
- **Treat patient data as sensitive.** `user_prompt.txt` and `Patients.xlsx` hold real patient records. Do not send them anywhere except the Anthropic API calls the scripts already make. Do not paste them into commits, issues, or logs.
- **Keep the Hebrew prompts in Hebrew.** Change their meaning only when asked.
- **Keep `guidelines_descriptions.json` in step with `NCCN_Guidlines/`.** Every guideline needs an entry with a matching filename.
- **Do not rename `NCCN_Guidlines/`** unless asked; update every reference if you do.
