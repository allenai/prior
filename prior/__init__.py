import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

# Literal was introduced in Python 3.8
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import requests
from attrs import define
from github import Github, GithubException

from .lock import LockEx

gh_auth_token: Optional[str] = None
"""The GitHub authentication token to use for requests."""


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
    dataset: str,
    revision: str = "main",
    entity: str = "allenai",
    config: Any = None,
    offline: bool = False,
) -> DatasetDict:
    """Load the dataset from the given revision.

    Args:
        dataset: The name of the dataset to load.
        revision: The git revision of the dataset to load. Can be specified as either
            a commit id sha, tag, or branch. If None, the latest commit to main
            will be used.
        entity: The github organization or username that has the dataset.
        config: Allows you to specify variants of a particular dataset (e.g., do you
            want the variant with a locobot or a different agent?).
        offline: If True, don't attempt to download the dataset from github.

    Returns:
        A DatasetDict containing the loaded dataset.
    """

    def get_cached_sha() -> Optional[str]:
        if os.path.exists(f"{dataset_dir}/cache"):
            with LockEx(f"{dataset_dir}/cache-lock"):
                with open(f"{dataset_dir}/cache", "r") as f:
                    cache = json.load(f)
                if revision in cache:
                    return cache[revision]
        return None

    dataset_dir = f"{os.environ['HOME']}/.prior/datasets/{entity}/{dataset}"
    os.makedirs(dataset_dir, exist_ok=True)
    start_dir = os.getcwd()

    sha: str
    cached_sha: Optional[str]
    token: str = ""

    if os.path.exists(f"{dataset_dir}/{revision}"):
        # If the dataset is already downloaded, use the cached sha.
        # NOTE: this will only occur if a commit id is passed in.
        # Otherwise, it tries to find the commit id.
        # NOTE: Not sure how it handles amend commits...
        sha = revision
    elif offline:
        cached_sha = get_cached_sha()
        if cached_sha is None or not os.path.isdir(f"{dataset_dir}/{cached_sha}"):
            raise ValueError(
                f"Offline dataset {dataset} is not downloaded "
                f"for revision {revision}. "
                f" cached_sha={cached_sha}, dataset_dir={dataset_dir}"
            )
        sha = cached_sha
        logging.debug(f"Using offline dataset {dataset} for revision {revision} with sha {sha}.")
    else:
        res = requests.get(
            f"https://api.github.com/repos/{entity}/{dataset}/commits?sha={revision}"
        )
        logging.debug(f"Getting status code {res.status_code} for {revision}")
        if res.status_code == 404 or res.status_code == 403:
            # Try using private repo.
            if (
                not os.path.exists(f"{os.environ['HOME']}/.git-credentials")
                and gh_auth_token is None
                and "GITHUB_TOKEN" not in os.environ
            ):
                # try using cache
                cached_sha = None
                if res.status_code == 403:
                    cached_sha = get_cached_sha()
                    if cached_sha is not None:
                        logging.debug("Exceeded API limit, using cached sha.")
                elif cached_sha is None:
                    raise Exception(
                        "Could not find dataset.\n"
                        "If you're using a private repo, "
                        "override the github auth token with:\n"
                        "    import prior\n"
                        "    prior.gh_auth_token = <token>\n"
                        "Alternatively, you can set the environment variable with:\n"
                        "    export GITHUB_TOKEN=<token>\n"
                        "from the command line."
                    )

            if gh_auth_token is not None:
                # Treats gh_auth_token as a string
                token = gh_auth_token.strip()  # type: ignore
            elif os.environ.get("GITHUB_TOKEN") is not None:
                token = os.environ["GITHUB_TOKEN"].strip()
            else:
                # look at ~/.git-credentials
                with open(f"{os.environ['HOME']}/.git-credentials", "r") as f:
                    tokens = f.read()
                token = next(token for token in tokens.split("\n") if token.endswith("github.com"))
                token = token.split(":")[2]
                token = token.split("@")[0]

            g = Github(token)
            repo = g.get_repo(f"{entity}/{dataset}")

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
        else:
            raise Exception(f"Unknown GitHub API status code: {res.status_code}")

        with LockEx(f"{dataset_dir}/cache-lock"):
            if os.path.exists(f"{dataset_dir}/cache"):
                with open(f"{dataset_dir}/cache", "r") as f:
                    cache = json.load(f)
            else:
                cache = {}
            with open(f"{dataset_dir}/cache", "w") as f:
                cache[revision] = sha
                json.dump(cache, f)

    # download the dataset
    dataset_path = f"{dataset_dir}/{sha}"
    if not os.path.exists(dataset_path):
        with LockEx(f"{dataset_dir}/lock"):
            logging.debug(
                f"Downloading dataset {dataset} at revision {revision} to {dataset_path}."
            )
            token_prefix = f"{token}@" if token else ""
            subprocess.run(
                args=[
                    "git",
                    "clone",
                    f"https://{token_prefix}github.com/{entity}/{dataset}.git",
                    dataset_path,
                ],
                stdout=subprocess.DEVNULL,
            )
            logging.debug(f"Downloaded dataset to {dataset_path}")
            # change the subprocess working directory to the dataset directory
            os.chdir(dataset_path)
            subprocess.run(
                args=["git", "checkout", sha],
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )
            logging.debug(f"Checked out {sha}")

    if os.path.exists(dataset_path):
        logging.debug(f"Found dataset {dataset} at revision {revision} in {dataset_path}.")
        os.chdir(dataset_path)

    out: Dict[str, Any] = {}

    # Necessary for GitHub Colab to work
    subprocess.run(
        args=["git lfs fetch origin && git lfs checkout".split(" ")],
        stderr=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )

    exec(open(f"{dataset_path}/main.py").read(), out)
    params = {}
    if config is not None:
        params["config"] = config
    out_dataset: DatasetDict = out["load_dataset"](**params)
    os.chdir(start_dir)
    return out_dataset
