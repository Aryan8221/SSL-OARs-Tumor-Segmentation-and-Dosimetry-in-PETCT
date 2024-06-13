import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Load the NIfTI file
nii_file_path = '/Users/aryanneizehbaz/Aryan8221/coding_projects/SSL-OARs-Tumor-Segmentation-in-PETCT/Dosimetry_Finetune/data_pre_processed/PET/PET11.nii'
nii_img = nib.load(nii_file_path)

# Get the data from the NIfTI file
nii_data = nii_img.get_fdata()

# Choose a slice to display (for 3D data)
slice_index = nii_data.shape[2] // 2  # Example: middle slice

# Plot the slice
plt.figure(figsize=(8, 8))
plt.imshow(nii_data[:, :, slice_index], cmap='gray')
plt.title(f'Slice {slice_index}')
plt.axis('off')
plt.show()