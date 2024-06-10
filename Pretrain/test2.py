# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from monai.data import (
    CacheDataset,
    DataLoader,
    Dataset,
    DistributedSampler,
    SmartCacheDataset,
    load_decathlon_datalist,
)
from monai.transforms import (
    AddChanneld,
    AsChannelFirstd,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)


def get_loader():
    num_workers = 4

    train_list = load_decathlon_datalist(
        '/Users/aryanneizehbaz/Aryan8221/coding_projects/SSL-OARs-Tumor-Segmentation-in-PETCT/Preprocessing/json_list.json', False, "training",
        base_dir='/Users/aryanneizehbaz/Aryan8221/coding_projects/SSL-OARs-Tumor-Segmentation-in-PETCT/Preprocessing/data'
    )
    val_list = load_decathlon_datalist(
        '/Users/aryanneizehbaz/Aryan8221/coding_projects/SSL-OARs-Tumor-Segmentation-in-PETCT/Preprocessing/json_list.json', False, "validation",
        base_dir='/Users/aryanneizehbaz/Aryan8221/coding_projects/SSL-OARs-Tumor-Segmentation-in-PETCT/Preprocessing/data'
    )

    print(
        "total number of trainin data: {}".format(len(train_list))
    )
    print(
        "total number of validation data: {}".format(len(val_list))
    )
    transforms_list = [
        LoadImaged(keys=["image"],image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        SpatialPadd(
            keys="image",
            spatial_size=[96, 96, 96],
        ),
        CropForegroundd(
            keys=["image"],
            source_key="image",
            k_divisible=[96, 96, 96],
        ),
        RandSpatialCropSamplesd(
            keys=["image"],
            roi_size=[96, 96, 96],
            num_samples=2,
            random_center=True,
            random_size=False,
        ),
        ToTensord(keys=["image"]),
    ]

    train_transforms = Compose(transforms_list)
    val_transforms = Compose(transforms_list)

    print("Using generic dataset")
    train_ds = Dataset(
        data=train_list, transform=train_transforms
    )

    train_sampler = None
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        num_workers=num_workers,
        sampler=train_sampler,
        drop_last=False,
    )
    # sample = next(iter(train_loader))

    val_ds = Dataset(data=val_list, transform=val_transforms)
    val_loader = DataLoader(
        val_ds,
        batch_size=2,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


train_loader, val_loader = get_loader()

print(dict(train_loader))
