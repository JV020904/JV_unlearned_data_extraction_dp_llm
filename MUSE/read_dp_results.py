import os
import json
import numpy as np

directory = "/nfs/home/jose/unlearned_data_extraction_llm_dp/checkpoint_updated/MUSE/clean_dp_forget"

label_map = {
    "True_1.0": "Post-Unlearning Extraction",
    "True_-1.0": "Pre-Unlearning Extraction",
    "True_-2.0": "Our Extraction",
}

files = []
for filename in sorted(os.listdir(directory)):
    if filename.endswith(".json") and "forget" in filename and "False_5.0" in filename:
        files.append(filename)

print(f"Found {len(files)} result files.\n")

for threshold in [0.9, 0.99]:
    print(f"Metric: A-ESR with Threshold {threshold}")

    for filename in files:
        with open(os.path.join(directory, filename), "r") as f:
            data = json.load(f)

        vals = np.array(data["rougeL_recall"])
        mean_value = np.mean(vals >= threshold)

        label = filename
        for key, name in label_map.items():
            if key in filename:
                label = name
                break

        print(f"{label}: {mean_value}")

    print()

print("Metric: Average RougeL(R)")

for filename in files:
    with open(os.path.join(directory, filename), "r") as f:
        data = json.load(f)

    vals = np.array(data["rougeL_recall"])

    label = filename
    for key, name in label_map.items():
        if key in filename:
            label = name
            break

    print(f"{label}: {vals.mean()}")

print("\nSanity check: Rouge-L distribution")

for filename in files:
    with open(os.path.join(directory, filename), "r") as f:
        data = json.load(f)

    vals = np.array(data["rougeL_recall"])

    label = filename
    for key, name in label_map.items():
        if key in filename:
            label = name
            break

    print(f"\n{label}")
    print(f"  n: {len(vals)}")
    print(f"  mean: {vals.mean()}")
    print(f"  max: {vals.max()}")
    print(f"  >= 0.5: {np.mean(vals >= 0.5)}")
    print(f"  >= 0.7: {np.mean(vals >= 0.7)}")
    print(f"  >= 0.9: {np.mean(vals >= 0.9)}")
    print(f"  >= 0.99: {np.mean(vals >= 0.99)}")
