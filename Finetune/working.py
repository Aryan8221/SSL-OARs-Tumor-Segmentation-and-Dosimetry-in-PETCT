from monai import data, transforms
import os
import torch
from monai.data import load_decathlon_datalist
from torch.utils.data import DataLoader


def get_loader():
    data_dir = '/Users/aryanneizehbaz/Aryan8221/coding_projects/SSL-OARs-Tumor-Segmentation-in-PETCT/Dosimetry_Finetune/masked_data'
    datalist_json = os.path.join(data_dir, 'masked_data.json')

    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=True),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            transforms.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"], image_only=True),
            transforms.EnsureChannelFirstd(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    # Load the datalists
    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)

    # Debugging: Print the length of datalist
    print(f"Training datalist length: {len(datalist)}")
    print(f"Validation datalist length: {len(val_files)}")

    # Create datasets
    train_ds = data.Dataset(data=datalist, transform=train_transform)
    val_ds = data.Dataset(data=val_files, transform=val_transform)

    # Debugging: Print the length of datasets
    print(f"Training dataset length: {len(train_ds)}")
    print(f"Validation dataset length: {len(val_ds)}")

    # Create DataLoaders without Sampler for non-distributed setup
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return [train_loader, val_loader]


loader = get_loader()

print(f"Length of training loader: {len(loader[0])}")  # Length of training loader
print(f"Length of validation loader: {len(loader[1])}")  # Length of validation loader
