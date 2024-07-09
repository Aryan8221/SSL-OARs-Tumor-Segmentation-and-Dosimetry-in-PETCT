import os
import re
import pandas as pd

# Directory containing log files
log_directory = "/Users/aryanneizehbaz/Aryan8221/coding_projects/SSL-OARs-Tumor-Segmentation-in-PETCT/Dosimetry_Finetune/loocv_runs"

# Initialize a dictionary to store the results
results = {"Fold": [], "Category": [], "Last R2 Score": []}

# Function to determine the category based on the file name
def determine_category(file_name):
    if "pet-ct-ssl" in file_name:
        return "pet-ct-ssl"
    elif "pet-ct" in file_name:
        return "pet-ct"
    elif "pet" in file_name:
        return "pet"
    else:
        return "unknown"

# Function to extract the fold number from the file name
def extract_fold_number(file_name):
    match = re.search(r'fold(\d+)', file_name)
    return int(match.group(1)) if match else float('inf')

# Function to process a log file and extract the last R2 score before "new best RMSE"
def process_log_file(log_file_path, log_file_name):
    r2_scores = []

    with open(log_file_path, "r") as file:
        log_contents = file.readlines()

    # Iterate through the lines to find "new best RMSE" and extract the previous line's R2 score
    for i in range(1, len(log_contents)):
        if "new best RMSE" in log_contents[i]:
            previous_line = log_contents[i - 1]
            match = re.search(r'R2: (\d+\.\d+)', previous_line)
            if match:
                r2_scores.append(float(match.group(1)))

    # If R2 scores were found, add the last one to the results
    last_r2_score = r2_scores[-1] if r2_scores else None
    results["Fold"].append(log_file_name)
    results["Category"].append(determine_category(log_file_name))
    results["Last R2 Score"].append(last_r2_score)

# Loop through each log file in the directory
for log_file in os.listdir(log_directory):
    log_file_path = os.path.join(log_directory, log_file)
    process_log_file(log_file_path, log_file)

# Create a DataFrame from the results
df = pd.DataFrame(results)

# Calculate averages for each category and add them to the DataFrame
averages = []
for category in ["pet-ct-ssl", "pet-ct", "pet"]:
    avg_r2 = df[df["Category"] == category]["Last R2 Score"].mean()
    averages.append({"Fold": "Average", "Category": category, "Last R2 Score": avg_r2})

# Append the averages to the DataFrame
df = pd.concat([df, pd.DataFrame(averages)], ignore_index=True)

# Sort the DataFrame by Category in the order of pet-ct-ssl, pet-ct, pet and then by fold number
category_order = pd.CategoricalDtype(["pet-ct-ssl", "pet-ct", "pet"], ordered=True)
df["Category"] = df["Category"].astype(category_order)
df["Fold Number"] = df["Fold"].apply(extract_fold_number)
df = df.sort_values(["Category", "Fold Number"]).reset_index(drop=True)

# Drop the temporary "Fold Number" column
df = df.drop(columns=["Fold Number"])

# Save the DataFrame to an Excel file
output_path = "/Users/aryanneizehbaz/Aryan8221/coding_projects/SSL-OARs-Tumor-Segmentation-in-PETCT/Dosimetry_Finetune/last_r2_scores.xlsx"
df.to_excel(output_path, index=False)

print(f"Results saved to {output_path}")
