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

import argparse
import os

from math import *
import nibabel as nib
import numpy as np
import torch
from utils.data_utils import get_loader
from utils.utils import resample_3d
import pandas as pd

from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR

parser = argparse.ArgumentParser(description="Swin UNETR dosimetry pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=2, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")  # Changed to 1 for dose map
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=4, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
parser.add_argument("--save", action="store_true", help="save dose map output")


def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = "./predictions/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    model = SwinUNETR(
        img_size=96,
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        use_checkpoint=args.use_checkpoint,
    )

    model_dict = torch.load(pretrained_pth, map_location=device)["state_dict"]

    # Print the keys to help debug
    print("Model state_dict keys:")
    print(model.state_dict().keys())
    print("\nLoaded state_dict keys:")
    print(model_dict.keys())

    # Filter the state_dict to only contain matching keys
    model_state_dict = model.state_dict()
    filtered_dict = {k: v for k, v in model_dict.items() if k in model_state_dict}
    model_state_dict.update(filtered_dict)
    model.load_state_dict(model_state_dict)

    model.eval()
    model.to(device)

    with torch.no_grad():
        mse_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
            )
            val_outputs = val_outputs.cpu().numpy()[0, 0]
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
            val_outputs = resample_3d(val_outputs, target_shape)
            mse = np.mean((val_outputs - val_labels) ** 2)
            mse_list_case.append(mse)
            if args.save:
                nib.save(
                    nib.Nifti1Image(val_outputs.astype(np.float32), original_affine), os.path.join(output_directory, img_name)
                )
        print("Overall Mean RMSE: {}".format(sqrt(np.mean(mse_list_case))))


if __name__ == "__main__":
    args = parser.parse_args()
    args.save = True
    main()
