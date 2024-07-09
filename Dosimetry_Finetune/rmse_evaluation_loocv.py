import os
import re
import pandas as pd

log_dir = 'loocv_runs'

rmse_pattern = re.compile(r'.*Training Finished.*, Best RMSE: ([0-9.]+)')

categories = {
    'pet': [],
    'pet-ct': [],
    'pet-ct-ssl': []
}

for log_file in os.listdir(log_dir):
    file_path = os.path.join(log_dir, log_file)
    if os.path.isfile(file_path):
        category = None
        if log_file.startswith('pet-ct-ssl'):
            category = 'pet-ct-ssl'
        elif log_file.startswith('pet-ct'):
            category = 'pet-ct'
        elif log_file.startswith('pet'):
            category = 'pet'

        if category:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in reversed(lines):
                    match = rmse_pattern.match(line)
                    if match:
                        rmse = float(match.group(1))
                        categories[category].append(rmse)
                        break

# Create a DataFrame to store the results
data = {
    'Category': [],
    'Fold': [],
    'Best RMSE': []
}

# Calculate average RMSE for each category
average_rmses = {}
for category, rmses in categories.items():
    if rmses:
        average_rmse = sum(rmses) / len(rmses)
        average_rmses[category] = average_rmse
        for fold, rmse in enumerate(rmses):
            data['Category'].append(category)
            data['Fold'].append(f'{category}-fold{fold}')
            data['Best RMSE'].append(rmse)
        # Adding the average RMSE for the category
        data['Category'].append(category)
        data['Fold'].append('Average')
        data['Best RMSE'].append(average_rmse)
    else:
        print(f'No RMSE values found for {category}')

print("-" * 20)

# Create a new structure to store the best RMSE and corresponding fold for each category
best_folds = {
    'pet': {},
    'pet-ct': {},
    'pet-ct-ssl': {}
}

for log_file in os.listdir(log_dir):
    file_path = os.path.join(log_dir, log_file)
    if os.path.isfile(file_path):
        category = None
        if log_file.startswith('pet-ct-ssl'):
            category = 'pet-ct-ssl'
        elif log_file.startswith('pet-ct'):
            category = 'pet-ct'
        elif log_file.startswith('pet'):
            category = 'pet'

        if category:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                for line in reversed(lines):
                    match = rmse_pattern.match(line)
                    if match:
                        rmse = float(match.group(1))
                        best_folds[category][log_file] = rmse
                        break

# Calculate and print the best RMSE for each category
for category, folds in best_folds.items():
    if folds:
        best_fold = min(folds, key=folds.get)
        best_rmse = folds[best_fold]
        data['Category'].append(category)
        data['Fold'].append(f'Best ({best_fold})')
        data['Best RMSE'].append(best_rmse)
        print(f'Best RMSE for {category}: {best_rmse:.6f} (Fold: {best_fold})')
    else:
        print(f'No RMSE values found for {category}')

# Convert the data to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to an Excel file
output_file = 'rmse_results_loocv.xlsx'
df.to_excel(output_file, index=False)

print(f"RMSE calculation complete. Results saved to {output_file}.")
