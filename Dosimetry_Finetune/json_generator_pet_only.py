import argparse
import os
import random
import json
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def load_files_in_directory(directory):
    return sorted(os.listdir(directory))

def generate_supervised_json(args):
    set_seed(42)

    pet_files = load_files_in_directory(os.path.join(args.path, "PET"))
    mask_files = load_files_in_directory(os.path.join(args.path, "Mask"))

    assert len(pet_files) == len(mask_files)

    cut_index = int(args.ratio * len(pet_files))

    zipped = list(zip(pet_files, mask_files))
    random.shuffle(zipped)
    pet_files, mask_files = zip(*zipped)

    folds_pet = np.array_split(pet_files, args.folds, axis=0)
    folds_mask = np.array_split(mask_files, args.folds, axis=0)

    for i in range(args.folds):
        pet_files_val, mask_files_val = folds_pet[i], folds_mask[i]
        pet_files_train = np.concatenate(folds_pet[:i] + folds_pet[i+1:], axis=0)
        mask_files_train = np.concatenate(folds_mask[:i] + folds_mask[i+1:], axis=0)

        print(
            f"# Training samples: {len(pet_files_train)}\t# Validation samples: {len(pet_files_val)}"
        )

        training = []
        validation = []

        for pet, mask in zip(pet_files_train, mask_files_train):
            if not (pet.replace("PET", "") == mask.replace("DML", "")):
                print(f"not aligned:\tpet:{pet}\tmask:{mask}")

            training.append({
                "image": f"./PET/{pet}",
                "label": f"./Mask/{mask}"
            })

        for pet, mask in zip(pet_files_val, mask_files_val):
            assert pet.replace("PET", "") == mask.replace("DML", "")
            validation.append({
                "image": f"./PET/{pet}",
                "label": f"./Mask/{mask}"
            })

        data = {"training": training, "validation": validation}

        with open(f"pet-fold{i}.json", "w") as f:
            json.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON for a dataset")
    parser.add_argument(
        "--path", default="dataset/dataset0", type=str, help="Path to the images"
    )
    parser.add_argument(
        "--json", default="jsons/dataset0.json", type=str, help="Path to the JSON output"
    )
    parser.add_argument(
        "--mode", type=str, help="Specify the data generation mode (ssl or sl)"
    )
    parser.add_argument(
        "--ratio", default=0.1, type=float, help="Ratio of validation data"
    )
    parser.add_argument(
        "--folds", default=1, type=int, help="Number of folds"
    )
    args = parser.parse_args()

    args.path = "data"
    args.ratio = 0.2
    args.folds = 5
    args.json = "folds"

    generate_supervised_json(args)
