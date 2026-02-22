# PRD: Analysis Top 20 Treatment Reasons v1

## Introduction

Create a new analysis script that uses LLMs to classify oncology patients into 3 treatment types and identify the top 20 reasons for treatment decisions across all patients. The script analyzes patient cases (approximately 480 patients), extracts treatment type and primary reason for each patient, then aggregates all reasons to identify the top 20 most significant reasons with explanations for why each was included, excluded, or merged.

## Goals

- Classify each patient into one of 3 treatment types: Chemo + immuno, immuno only, or Immuno + chemo reduce dose
- Extract a single primary reason for the treatment decision for each patient
- Aggregate all patient reasons and identify the top 20 most significant reasons
- Provide explanations for why reasons were excluded from top 20 or merged with similar reasons
- Support 3 LLM providers: Sonnet 4.5, OpenAI GPT5, and OpenAI o1
- Validate patient counts match input totals and log discrepancies
- Handle edge cases gracefully (invalid treatment types, uncategorized patients)

## User Stories

### US-001: Create new analysis script with CLI argument for LLM provider

**Description:** As a developer, I need to create a new Python script that accepts an LLM provider as a CLI argument so I can run the analysis with different models.

**Acceptance Criteria:**

- [ ] Create `analysis_top_20_reasons_v1.py` script
- [ ] Script accepts `--provider` CLI argument (default: `azure_openai`)
- [ ] Valid provider values: `azure_openai`, `bedrock`, `anthropic`
- [ ] Script prints the selected provider at startup
- [ ] Error if invalid provider is specified

### US-002: Read input CSV and validate structure

**Description:** As a developer, I need to read the cases.csv file and validate its structure matches expected columns.

**Acceptance Criteria:**

- [ ] Read from `./cases.csv` (same as existing analysis scripts)
- [ ] Validate required columns exist: `PatId`, `Current_Disease`, `Summary_Conclusions`, `Recommendations`
- [ ] Print warning if column names don't match exactly but required columns exist
- [ ] Exit with error if required columns are missing
- [ ] Count total patients and print at startup

### US-003: Implement first LLM call for patient classification

**Description:** As a developer, I need to implement the first LLM call that classifies each patient into a treatment type and provides a single primary reason.

**Acceptance Criteria:**

- [ ] Use Hebrew system prompt: "אתה רופא אונקולוג..."
- [ ] Prompt asks for exactly one treatment type from the 3 options
- [ ] Prompt asks for a single primary reason (not multiple)
- [ ] Patient data included: Current_Disease, Summary_Conclusions, Recommendations
- [ ] Use `invoke_llm()` from llm_client.py with max_tokens=1000, temperature=0.0
- [ ] Store LLM response for later processing

### US-004: Parse and normalize LLM response

**Description:** As a developer, I need to parse the LLM response to extract treatment type and primary reason, then normalize the treatment type to standard English strings.

**Acceptance Criteria:**

- [ ] Extract treatment type from LLM response (free text extraction)
- [ ] Extract primary reason from LLM response (free text)
- [ ] Normalize treatment type to one of: "Chemo + immuno", "immuno only", "Immuno + chemo reduce dose"
- [ ] Handle flexible Hebrew/English input for treatment types
- [ ] Log treatment type normalization decisions to console

### US-005: Handle invalid treatment types with uncategorized logging

**Description:** As a developer, I need to handle cases where the LLM returns a treatment type that doesn't match the 3 valid categories.

**Acceptance Criteria:**

- [ ] If treatment type cannot be normalized, classify as "Chemo + immuno"
- [ ] Log the patient to `uncategorized.csv` file
- [ ] uncategorized.csv columns: PatId, original_response, normalized_as
- [ ] Print warning to console for each uncategorized patient
- [ ] Continue processing subsequent patients

### US-006: Save patient results incrementally

**Description:** As a developer, I need to save patient results after each patient is processed to prevent data loss on interruption.

**Acceptance Criteria:**

- [ ] Create output file `analysis_v1_results.csv` with headers before processing
- [ ] Append each patient's result immediately after processing
- [ ] Output columns: PatId, treatment_type, primary_reason
- [ ] If file exists, append mode (don't overwrite)
- [ ] Handle file write errors gracefully

### US-007: Collect all patient reasons for top-20 analysis

**Description:** As a developer, I need to collect all patient reasons after processing all patients for the second LLM call.

**Acceptance Criteria:**

- [ ] Store all patient reasons in a list/array during processing
- [ ] Each reason includes: PatId, reason_text
- [ ] Print count of total reasons collected after processing all patients
- [ ] Verify count matches number of patients processed

### US-008: Implement second LLM call for top-20 reasons selection

**Description:** As a developer, I need to implement the second LLM call that receives all patient reasons and selects the top 20 with explanations.

**Acceptance Criteria:**

- [ ] Use Hebrew prompt: "אתה מקבל עכשיו את הסיבה..."
- [ ] Send all patient reasons to LLM (approximately 480 reasons)
- [ ] Ask LLM to select top 20 reasons
- [ ] Allow LLM to merge similar reasons (e.g., "PDL1 חזק" and "PDL1 high")
- [ ] Ask LLM to explain for each reason why included, excluded, or merged
- [ ] Use same LLM provider as patient analysis
- [ ] Use max_tokens=4000, temperature=0.0

### US-009: Parse top-20 LLM response

**Description:** As a developer, I need to parse the top-20 LLM response to extract the structured data about reasons.

**Acceptance Criteria:**

- [ ] Extract top 20 reason names/descriptions
- [ ] Extract patient count for each top reason
- [ ] Extract explanation for why each was included
- [ ] For each original patient reason, determine:
  - Which top-20 group it was merged with (if applicable)
  - Why it was excluded (if not in top 20)
- [ ] Handle cases where LLM returns fewer than 20 reasons

### US-010: Create not_in_top_20.csv for excluded reasons

**Description:** As a developer, I need to create a separate CSV file that logs patients whose reasons were not included in the top 20, with explanations.

**Acceptance Criteria:**

- [ ] Create `not_in_top_20.csv` file
- [ ] Columns: PatId, primary_reason, exclusion_reason
- [ ] Include all patients whose primary reason is not in the top 20
- [ ] Populate exclusion_reason with LLM's explanation
- [ ] Create file after second LLM call completes

### US-011: Append top-20 reasons to output CSV

**Description:** As a developer, I need to append the top-20 reasons and their metadata to the main output CSV.

**Acceptance Criteria:**

- [ ] Append top-20 reasons to `analysis_v1_results.csv`
- [ ] Format: Each top reason as a row with special identifier (e.g., PatId = "TOP_REASON_N")
- [ ] Columns: PatId, reason_name, patient_count, explanation
- [ ] Include all 20 reasons (or fewer if LLM returned less)
- [ ] Print message to console indicating top reasons appended

### US-012: Validate patient count totals

**Description:** As a developer, I need to verify that the sum of patients across treatment types matches the total input and log discrepancies.

**Acceptance Criteria:**

- [ ] Count patients in each treatment type category
- [ ] Sum counts and compare to total input patients
- [ ] If mismatch: print warning to console and log to stderr
- [ ] Warning message includes both counts for comparison
- [ ] Do not stop execution on mismatch

### US-013: Implement rate limiting between requests

**Description:** As a developer, I need to implement rate limiting to respect API constraints for Azure OpenAI (and optionally other providers).

**Acceptance Criteria:**

- [ ] Use 31-second delay between patient LLM calls for Azure OpenAI
- [ ] No delay for Anthropic provider (ANTHROPIC_NO_RATE_LIMIT logic)
- [ ] Print countdown message before each delay
- [ ] Apply delay after each patient is processed
- [ ] No delay before first patient

### US-014: Add treatment type normalization rules

**Description:** As a developer, I need to implement flexible normalization rules for the 3 treatment types to handle various Hebrew/English inputs.

**Acceptance Criteria:**

- [ ] Define normalization mappings for "Chemo + immuno":
  - Hebrew: "כימותרפיה ואימונותרפיה", "כימו ואימונו", "אימונו וכימו"
  - English variants: "Chemo+immuno", "Chemotherapy + immunotherapy"
- [ ] Define normalization mappings for "immuno only":
  - Hebrew: "אימונותרפיה בלבד", "רק אימונו", "אימונו בלבד"
  - English variants: "Immunotherapy only", "Immuno alone"
- [ ] Define normalization mappings for "Immuno + chemo reduce dose":
  - Hebrew: "אימונו וכימו במינון מופחת", "כימו מופחת ואימונו"
  - English variants: "Reduced dose chemo + immuno", "Chemo reduced dose + immunotherapy"
- [ ] Apply fuzzy matching for close variations
- [ ] Log each normalization decision to console

### US-015: Handle LLM API errors gracefully

**Description:** As a developer, I need to handle LLM API errors to prevent script failure and allow recovery.

**Acceptance Criteria:**

- [ ] Detect LLM responses starting with "ERROR:"
- [ ] Log error details to console
- [ ] Mark patient with error: treatment_type = "ERROR", primary_reason = error_message
- [ ] Continue processing next patient
- [ ] Track error count and print summary at end
- [ ] Retry option for transient errors (optional, future enhancement)

## Functional Requirements

- FR-1: The system must read patient data from `./cases.csv` with columns: PatId, Current_Disease, Summary_Conclusions, Recommendations
- FR-2: The system must accept a `--provider` CLI argument to select LLM provider (default: `azure_openai`)
- FR-3: The system must invoke the first LLM for each patient with Hebrew prompt requesting treatment type and single primary reason
- FR-4: The system must normalize treatment types to: "Chemo + immuno", "immuno only", or "Immuno + chemo reduce dose"
- FR-5: The system must log patients with unclassifiable treatment types to `uncategorized.csv` and classify as "Chemo + immuno"
- FR-6: The system must save patient results incrementally to `analysis_v1_results.csv` after each patient is processed
- FR-7: The system must collect all patient reasons and send them to a second LLM call
- FR-8: The system must invoke the second LLM with Hebrew prompt to select top 20 reasons from all patient reasons
- FR-9: The system must parse the top-20 response to extract reason names, patient counts, and explanations
- FR-10: The system must create `not_in_top_20.csv` with patients whose reasons were excluded, including exclusion explanations
- FR-11: The system must append top-20 reasons to `analysis_v1_results.csv` with special identifier rows
- FR-12: The system must validate that sum of patients across treatment types matches input total, logging mismatch to stderr if found
- FR-13: The system must implement 31-second delay between Azure OpenAI LLM calls, no delay for Anthropic
- FR-14: The system must use the same LLM provider for both patient analysis and top-20 selection
- FR-15: The system must print progress messages: processing record N/X, waiting message, completion status

## Non-Goals (Out of Scope)

- No automatic retry for failed LLM calls (future enhancement)
- No parallel processing of patients (sequential processing only)
- No web UI or visualization of results
- No comparison between different LLM providers in a single run (run separately)
- No translation of Hebrew responses to English
- No constraint on reason terminology (free text allowed)
- No statistical analysis or metrics beyond count validation
- No caching of LLM responses
- No batching of multiple patients in a single LLM call

## Design Considerations

### Existing Code to Reuse
- `llm_client.py`: Use `invoke_llm()` function for all LLM calls
- `analysis_per_case.py`: Reference for CSV reading/writing patterns, error handling
- `analysis_per_case_2.py`: Reference for structured output patterns

### Hebrew System Prompt (First LLM Call)
```
אתה רופא אונקולוג, אתה צריך לציין את הסיבות וההגיון שהובילו את הרופא להחלטה על הסוג טיפול
סוג טיפול יכול להיות אחד מתוך שלושה - אימונו בלבד, אימונו וכימותרפיה או אימונו וכימותרפיה במינון מופחת.
עבור כל מטופל עלייך לציין את הסיבה למה הרופא בחר בסוג טיפול זה
```

### Hebrew User Prompt Format
```
כך סיכם הרופא את המקרה:
{Current_Disease}

זה מה שהחליט הרופא:
{Recommendations} {Summary_Conclusions}
```

### Hebrew Prompt (Second LLM Call)
```
אתה מקבל עכשיו את הסיבה שבגינה רופא בחר להחליט על סוג טיפול, עלייך לבחור טופ 20 סיבות מכל רשימת הסיבות ועבור כל סיבה ברשימה להסביר למה הורדת או למה השארת את הסיבה הזו
```

### Output File Structure

**analysis_v1_results.csv:**
```csv
PatId,treatment_type,primary_reason
51164184,immuno only,PDL1 גבוה
51164185,Chemo + immuno,מחלת רקע כבד
...
TOP_REASON_1,PDL1 גבוה/חזק,125,ראשית במשמעות - ביטוי PD-L1 הוא הפקטור החזק ביותר
TOP_REASON_2,מחלת רקע כבדית,45,מניעת תופעות לוואי של כימותרפיה
...
```

**uncategorized.csv:**
```csv
PatId,original_response,normalized_as
51164200,טיפול פליאטיבי,Chemo + immuno
```

**not_in_top_20.csv:**
```csv
PatId,primary_reason,exclusion_reason
51164200,סוכרת לא מאוזנת,סיבה נדירה שהופיעה ב-2 מטופלים בלבד, לא משמעותית לטופ 20
```

## Technical Considerations

### LLM Client Integration
- Use `invoke_llm(system_prompt, user_prompt_text, max_tokens, temperature, provider_override)` from llm_client.py
- First call: max_tokens=1000, temperature=0.0
- Second call: max_tokens=4000, temperature=0.0
- Provider determined by CLI argument, passed to invoke_llm as provider_override

### Rate Limiting
- Borrow from existing analysis scripts: REQUEST_DELAY_SECONDS = 31
- ANTHROPIC_NO_RATE_LIMIT logic for Anthropic provider
- Print countdown message before each delay

### Error Handling
- LLM errors starting with "ERROR:" should be handled gracefully
- Continue processing next patient on error
- Track and report error count at end

### CSV Writing
- Use Python csv.DictWriter for structured output
- Incremental writing: open file in 'a' mode after first write
- Use UTF-8 encoding for Hebrew characters

### Treatment Type Normalization
- Implement flexible mapping with fuzzy matching
- Log each normalization decision
- Default to "Chemo + immuno" for unclassifiable responses

### Environment Variables
- LLM_PROVIDER can be set via .env or CLI argument
- CLI argument takes precedence over .env

## Success Metrics

- All patients (480) processed without script failure
- LLM responses parsed successfully for 95%+ of patients
- Top-20 reasons output includes patient counts for each reason
- not_in_top_20.csv is created with explanations for excluded reasons
- Patient count validation passes (or discrepancy is logged)
- Output files are valid CSV with UTF-8 encoding
- Script can be interrupted and resumed (incremental saves)

## Open Questions

- Should we implement a retry mechanism for transient LLM API errors?
- Should the treatment type normalization rules be configurable (e.g., via JSON file)?
- Should we add a progress bar instead of simple text messages?
- Should we add an option to skip rate limiting for testing purposes?
- Should we validate that sum of patient counts in top-20 reasons equals total patients?
