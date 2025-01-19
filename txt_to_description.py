import anthropic
import json
import os
import re

API_KEY = "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"

client = anthropic.Anthropic(api_key=API_KEY)

# Initialize results dictionary
results = {}

# Process each file individually
guideline_files = [f for f in os.listdir('NCCN_Guidlines/') if f.endswith('.txt')]
total_files = len(guideline_files)

for idx, filename in enumerate(guideline_files, 1):
    print(f"Processing file {idx}/{total_files}: {filename}")

    # Read file preview
    with open(os.path.join('NCCN_Guidlines', filename), 'r', encoding='utf-8') as file:
        lines = file.readlines()[:300]
        content = re.sub(r' +', ' ', ' '.join(lines))

    # Create prompt for single file
    prompt = (
        "Provide a 50-word description of the disease discussed in this medical guideline, "
        "focusing only on medical concepts. Try to fill as much of the 50 words as possible "
        "with medical terms. Only output the description, nothing else.\n\n"
        "Here is the document:\n\n"
    )
    prompt += content

    # Make API call
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=100,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    # Store result
    results[filename] = message.content[0].text.strip()

    print(f"Completed {filename}\n")

# Print final JSON
print("\nFinal Results:")
print(json.dumps(results, indent=2))

# Write results to JSON file
with open('./guidelines_descriptions.json', 'w') as f:
    json.dump(results, f, indent=2)
