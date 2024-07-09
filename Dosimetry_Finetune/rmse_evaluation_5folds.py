import os
import re
import pandas as pd

# Define the path to the folder containing the log files
folder_path = 'results-rmse-r2-mae-smape'

# Define a regex pattern to match the required lines
pattern = re.compile(r".*new best RMSE \((\d+\.\d+) --> (\d+\.\d+)\).*")

# Dictionaries to store the RMSE values for each category and track the best fold
rmse_values = {
    'pet-ct': {},
    'pet-ct-ssl': {},
    'pet': {}
}
best_folds = {
    'pet-ct': ('', float('inf')),
    'pet-ct-ssl': ('', float('inf')),
    'pet': ('', float('inf'))
}

# Function to process each log file and extract the required RMSE values
def process_log_file(file_path):
    category = None
    fold = None
    if 'pet-ct-' in file_path and 'pet-ct-ssl-' not in file_path:
        category = 'pet-ct'
        fold = re.search(r'pet-ct-fold(\d+)', file_path)
    elif 'pet-ct-ssl-' in file_path:
        category = 'pet-ct-ssl'
        fold = re.search(r'pet-ct-ssl-fold(\d+)', file_path)
    elif 'pet-' in file_path and 'pet-ct-' not in file_path:
        category = 'pet'
        fold = re.search(r'pet-fold(\d+)', file_path)

    if category and fold:
        fold = int(fold.group(1))
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Extract the last matching line
        last_match = None
        for line in lines:
            match = pattern.match(line)
            if match:
                last_match = match

        # Add the last match value to the corresponding category list
        if last_match:
            rmse = float(last_match.group(2))
            rmse_values[category][fold] = rmse
            # Update the best fold if this RMSE is lower
            if rmse < best_folds[category][1]:
                best_folds[category] = (f'{category}-fold{fold}', rmse)

# Iterate over each file in the folder and apply the processing function
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    process_log_file(file_path)

# Function to calculate average RMSE for each category
def calculate_average_rmse(rmse_dict):
    if not rmse_dict:
        return None
    return sum(rmse_dict.values()) / len(rmse_dict)

# Create a DataFrame to store the results
data = {
    'Category': [],
    'Fold': [],
    'Best RMSE': []
}

# Calculate and store RMSE statistics for each category
for category, values in rmse_values.items():
    avg_rmse = calculate_average_rmse(values)
    best_fold, best_rmse = best_folds[category]
    if avg_rmse is not None:
        for fold, rmse in sorted(values.items()):
            data['Category'].append(category)
            data['Fold'].append(f'{category}-fold{fold}')
            data['Best RMSE'].append(rmse)
        # Adding the average RMSE for the category
        data['Category'].append(category)
        data['Fold'].append('Average')
        data['Best RMSE'].append(avg_rmse)
    else:
        print(f"Category: {category} has no valid RMSE data")

# Add the best RMSE for each category
for category, (best_fold, best_rmse) in best_folds.items():
    data['Category'].append(category)
    data['Fold'].append(f'Best ({best_fold})')
    data['Best RMSE'].append(best_rmse)

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
output_file = 'rmse_results.xlsx'
df.to_excel(output_file, index=False)

print(f"RMSE calculation complete. Results saved to {output_file}.")
