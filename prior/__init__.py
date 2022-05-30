import logging
import os
import subprocess
from typing import Any, Dict, List, Literal, Optional

import requests
from attrs import define
from github import Github, GithubException


@define
class Dataset:
    data: List[Any]
    """The entries in the dataset."""

    dataset: str
    """The split of the dataset."""

    split: Literal["train", "val", "test"]
    """The split of the dataset."""

    def __iter__(self):
        """Return an iterator over the dataset."""
        for item in self.data:
            yield item

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.data)

    def __getitem__(self, index: int) -> Any:
        """Return the item at the given index."""
        return self.data[index]

    def __repr__(self):
        """Return a string representation of the dataset."""
        return (
            "DatasetSplit(\n"
            f"    dataset={self.dataset},\n"
            f"    size={len(self.data)},\n"
            f"    split={self.split}\n"
            ")"
        )


@define
class DatasetDict:
    train: Optional[Dataset] = None
    val: Optional[Dataset] = None
    test: Optional[Dataset] = None

    def __getitem__(self, key: str) -> Any:
        if key == "train":
            if self.train is None:
                raise KeyError(key)
            return self.train
        elif key == "val":
            if self.val is None:
                raise KeyError(key)
            return self.val
        elif key == "test":
            if self.test is None:
                raise KeyError(key)
            return self.test


def load_dataset(
    dataset: str, revision: Optional[str] = None, entity: str = "allenai"
) -> DatasetDict:
    """Load the dataset from the given revision.

    Args:
        dataset: The name of the dataset to load.
        revision: The git revision of the dataset to load. Can be specified as either
            a commit id sha, tag, or branch. If None, the latest commit to main
            will be used.
        entity: The github organization or username that has the dataset.

    Returns:
        A DatasetDict containing the loaded dataset.
    """

    start_dir = os.getcwd()
    res = requests.get(f"https://api.github.com/repos/allenai/{dataset}/commits")
    if res.status_code == 404:
        # Try using private repo.
        if not os.path.exists(f"{os.environ['HOME']}/.git-credentials"):
            raise Exception(
                "Could not find ~/.git-credentials. "
                "Please make sure you're logged into GitHub with the following command:\n"
                "    git config --global credential.helper store"
            )

        with open(f"{os.environ['HOME']}/.git-credentials", "r") as f:
            tokens = f.read()
        token = next(token for token in tokens.split("\n") if token.endswith("github.com"))
        token = token.split(":")[2]
        token = token.split("@")[0]

        g = Github(token)
        repo = g.get_repo(f"{entity}/{dataset}")

        # main sha
        sha: str
        if revision is None:
            # get the latest commit
            sha = repo.get_branch("main").commit.sha
        elif any(revision == branch.name for branch in repo.get_branches()):
            # if revision is a branch name, get the commit_id of the branch
            sha = repo.get_branch(revision).commit.sha
        elif any(revision == tag.name for tag in repo.get_tags()):
            # if revision is a tag, get the commit_id of the tag
            sha = repo.get_tag(revision).commit.sha
        else:
            # if revision is a commit_id, use it
            sha = revision

        # make sure the commit_id is valid
        try:
            repo.get_commit(sha)
        except GithubException:
            raise GithubException(
                f"Could not find revision={revision} in dataset={entity}/{dataset}."
                " Please pass a valid commit_id sha, branch name, or tag."
            )

        # download the dataset
        dataset_dir = f"{os.environ['HOME']}/.prior/datasets/{dataset}"
        dataset_path = f"{dataset_dir}/{sha}"
        if os.path.exists(dataset_path):
            logging.info(f"Found dataset {dataset} at revision {revision} in {dataset_path}.")
        else:
            logging.info(f"Downloading dataset {dataset} at revision {revision} to {dataset_path}.")
            os.makedirs(dataset_dir, exist_ok=True)
            subprocess.run(
                args=["git", "clone", f"https://github.com/allenai/{dataset}.git", dataset_path],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            # change the subprocess working directory to the dataset directory
            os.chdir(dataset_dir)
            subprocess.run(
                args=["git", "checkout", sha],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

        os.chdir(dataset_path)

        out: Dict[str, Any] = {}
        exec(open(f"{dataset_path}/main.py").read(), out)
        out_dataset: DatasetDict = out["load_dataset"]()
        os.chdir(start_dir)
        return out_dataset
    raise NotImplementedError("Dataset not .")
