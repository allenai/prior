import gzip
import json
import os

from prior import Dataset, DatasetDict


def load_dataset(revision: str = None) -> DatasetDict:
    # download the data to ~/.prior/datasets/houses/revision/
    # if os.path.exists(f"{os.environ['HOME']}/.prior/datasets/houses/{revision}/houses.json.gz"):
    #     print(f"Found dataset {revision}")
    # else:
    #     # show progress bar for download
    #     pass

    # github api
    # get the current directory

    with gzip.open(f"{os.path.dirname(__file__)}/houses.json.gz", "rb") as f:
        houses = json.loads(f.read().decode("utf-8"))

    return DatasetDict(
        train=Dataset(data=houses["train"], dataset="houses", split="train"),
        val=Dataset(data=houses["validation"], dataset="houses", split="val"),
        test=Dataset(data=houses["test"], dataset="houses", split="test"),
    )


# import datasets
# import pickle

# dataset = datasets.load_dataset("allenai/houses", use_auth_token=True)

# houses = {
#     "train": [],
#     "validation": [],
#     "test": [],
# }
# for split in ["train", "validation", "test"]:
#     for house_entry in dataset[split]:
#         house = pickle.loads(house_entry["house"])
#         houses[split].append(house)
#     print(f"{split} {len(dataset[split])}")

# # save houses as a gzip json file
# import json
# with gzip.open("houses.json.gz", "wb") as f:
#     f.write(json.dumps(houses).encode("utf-8"))

# read houses from gzip json file
