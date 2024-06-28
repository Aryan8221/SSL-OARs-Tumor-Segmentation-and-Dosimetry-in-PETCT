import os
import re

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

for category, rmses in categories.items():
    if rmses:
        average_rmse = sum(rmses) / len(rmses)
        print(f'Average RMSE for {category}: {average_rmse:.6f}')
    else:
        print(f'No RMSE values found for {category}')

print("-" * 20)

categories = {
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
                        categories[category][log_file] = rmse
                        break

for category, folds in categories.items():
    if folds:
        best_fold = min(folds, key=folds.get)
        best_rmse = folds[best_fold]
        print(f'Best RMSE for {category}: {best_rmse:.6f} (Fold: {best_fold})')
    else:
        print(f'No RMSE values found for {category}')
