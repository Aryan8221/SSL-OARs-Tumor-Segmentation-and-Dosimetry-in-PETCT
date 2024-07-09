import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Load the NIfTI file
nii_file_path = '/Users/aryanneizehbaz/Aryan8221/coding_projects/SSL-OARs-Tumor-Segmentation-in-PETCT/Dosimetry_Finetune/predictions/ADRM41N.nii.gz'
nii_img = nib.load(nii_file_path)

# Get voxel size
voxel_size = nii_img.header.get_zooms()
print(f"Voxel size: {voxel_size}")

# Get the data from the NIfTI file
nii_data = nii_img.get_fdata()

# Calculate scalar range
scalar_range = (np.min(nii_data), np.max(nii_data))
print(f"Scalar range: {scalar_range}")

# Choose a slice to display (for 3D data)
slice_index = nii_data.shape[2] // 2  # Example: middle slice

# Plot the slice
plt.figure(figsize=(8, 8))
plt.imshow(nii_data[:, :, slice_index], cmap='gray')
plt.title(f'Slice {slice_index}\nVoxel size: {voxel_size}\nScalar range: {scalar_range}')
plt.axis('off')
plt.show()

# Get affine matrix
affine = nii_img.affine

# Print image origin
image_origin = affine[:3, 3]
print(f"Image Origin: {image_origin}")
