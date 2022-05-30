import logging
import os
import subprocess
from platform import system
from typing import Any, Dict, List, Literal, Optional

import requests
from attrs import define
from github import Github, GithubException

if system() == "Windows":

    class fcntl:
        LOCK_UN = 0
        LOCK_SH = 0
        LOCK_NB = 0
        LOCK_EX = 0

        @staticmethod
        def fcntl(fd, op, arg=0):
            return 0

        @staticmethod
        def ioctl(fd, op, arg=0, mutable_flag=True):
            return 0 if mutable_flag else ""

        @staticmethod
        def flock(fd, op):
            return

        @staticmethod
        def lockf(fd, operation, length=0, start=0, whence=0):
            return

else:
    import fcntl


class Lock:
    def __init__(self, target, mode):
        self._lock_file_path = target + ".lock"
        self._lock_file = os.open(self._lock_file_path, os.O_RDWR | os.O_CREAT)
        self.mode = mode

    def lock(self):
        fcntl.lockf(self._lock_file, self.mode)

    def unlock(self):
        fcntl.lockf(self._lock_file, fcntl.LOCK_UN)
        os.close(self._lock_file)

    def unlink(self):
        os.unlink(self._lock_file_path)

    def __enter__(self):
        self.lock()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.unlock()


class LockEx(Lock):
    def __init__(self, target, blocking=True):
        mode = fcntl.LOCK_EX
        if not blocking:
            mode |= fcntl.LOCK_NB

        super().__init__(target, mode)


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
            "Dataset(\n"
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

    def __getitem__(self, key: str) -> Dataset:
        """Return the dataset with the given split."""
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
        else:
            raise KeyError(key)


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

    with LockEx(f"{entity}/{dataset}"):
        start_dir = os.getcwd()
        res = requests.get(
            f"https://api.github.com/repos/{entity}/{dataset}/commits?sha={revision}"
        )
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
            else:
                # if revision is a commit_id, branch, or tag use it
                try:
                    sha = repo.get_commits(sha=revision)[0].sha
                except GithubException:
                    raise GithubException(
                        f"Could not find revision={revision} in dataset={entity}/{dataset}."
                        " Please pass a valid commit_id sha, branch name, or tag."
                    )
        elif res.status_code == 200:
            sha = res.json()[0]["sha"]
        elif res.status_code == 403:
            # TODO: try using cached sha
            raise Exception("GitHub API rate limit exceeded. Please wait a minute and try again.")
        else:
            raise Exception(f"Unknown GitHub API status code: {res.status_code}")

        # download the dataset
        dataset_dir = f"{os.environ['HOME']}/.prior/datasets/{entity}/{dataset}"
        dataset_path = f"{dataset_dir}/{sha}"
        if os.path.exists(dataset_path):
            logging.info(f"Found dataset {dataset} at revision {revision} in {dataset_path}.")
            os.chdir(dataset_path)
        else:
            logging.info(f"Downloading dataset {dataset} at revision {revision} to {dataset_path}.")
            os.makedirs(dataset_dir, exist_ok=True)
            subprocess.run(
                args=["git", "clone", f"https://github.com/{entity}/{dataset}.git", dataset_path],
                stdout=subprocess.DEVNULL,
            )
            # change the subprocess working directory to the dataset directory
            os.chdir(dataset_path)
            subprocess.run(
                args=["git", "checkout", sha],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

        out: Dict[str, Any] = {}
        exec(open(f"{dataset_path}/main.py").read(), out)
        out_dataset: DatasetDict = out["load_dataset"]()
        os.chdir(start_dir)
        return out_dataset
