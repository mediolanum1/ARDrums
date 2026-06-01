import json
from collections import defaultdict

# Input and output file names
INPUT_FILE = "results/rhythm_session_2_gunak_drum2.json"
OUTPUT_FILE = "results/averaged_results_drum2.json"


def average_dicts(dicts):
    """
    Recursively averages numeric values in nested dictionaries.
    """
    result = {}

    keys = dicts[0].keys()

    for key in keys:
        values = [d[key] for d in dicts]

        # Nested dictionary
        if isinstance(values[0], dict):
            result[key] = average_dicts(values)

        # Numeric value
        elif isinstance(values[0], (int, float)):
            result[key] = round(sum(values) / len(values), 4)

        # Non-numeric (keep first value)
        else:
            result[key] = values[0]

    return result


def main():
    # Load JSON data
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    # Group phases by label (Slow / Fast)
    grouped = defaultdict(list)

    for sample in data:
        for phase in sample["phases"]:
            label = phase["label"]
            grouped[label].append(phase)

    # Compute averages
    final_results = {}

    for label, phases in grouped.items():
        final_results[label] = average_dicts(phases)

    # Write output JSON
    with open(OUTPUT_FILE, "w") as f:
        json.dump(final_results, f, indent=4)

    print(f"Averaged results written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()