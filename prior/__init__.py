import glob
import json
import logging
import os
import platform
import stat
import subprocess
import zipfile
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

DATASET_DIR = f"{os.environ['HOME']}/.prior/datasets"

gh_auth_token: Optional[str] = None
"""The GitHub authentication token to use for requests."""


_GIT_LFS_DOWNLOAD_TEMPLATE = (
    "https://github.com/git-lfs/git-lfs/releases/download/v3.2.0/git-lfs-{os}-{arch}-v3.2.0.tar.gz"
)
_LFS_FILE_TO_SHA256 = {
    "git-lfs-darwin-amd64-v3.2.0.zip": "c48c6a0c21d6fd286e54154fedae109bca9886caf520336cbdbbde1f209d8aff",
    "git-lfs-darwin-arm64-v3.2.0.zip": "bf0fbe944e2543cacca74749476ff3671dff178b5853489c1ca92a2d1b04118e",
    "git-lfs-freebsd-386-v3.2.0.tar.gz": "66ca0f662eeaefa2c191577f54d7d2797063f7f4e44c9130cf7186d8372df595",
    "git-lfs-freebsd-amd64-v3.2.0.tar.gz": "776b41b526f1c879b2a106780c735f58c85b79bf97a835140d4c1aefc8c935b6",
    "git-lfs-linux-386-v3.2.0.tar.gz": "73895460f9b3e213d10fb23948680681ab3e5f92e2fb0a74eb7830f6227a244e",
    "git-lfs-linux-amd64-v3.2.0.tar.gz": "d6730b8036d9d99f872752489a331995930fec17b61c87c7af1945c65a482a50",
    "git-lfs-linux-arm-v3.2.0.tar.gz": "3273b189fea5a403a2b6ab469071326ae4d97cb298364aa25e3b7b0e80340bad",
    "git-lfs-linux-arm64-v3.2.0.tar.gz": "8186f0c0f69c30b55863d698e0a20cf79447a81df006b88221c2033d1e893638",
    "git-lfs-linux-ppc64le-v3.2.0.tar.gz": "ff1eeaddde5d964d10ce607f039154fe033073f43b8ff5e7f4eb407293fe1be3",
    "git-lfs-linux-s390x-v3.2.0.tar.gz": "16556f0b2e1097a69e75a6e1bcabfa7bfd2e7ee9b02fe6e5414e1038a223ab97",
    "git-lfs-v3.2.0.tar.gz": "f8e6bbe043b97db8a5c16da7289e149a3fed9f4d4f11cffcc6e517c7870cd9e5",
}


def _get_git_lfs_cmd():
    # Trying to install git-lfs locally to $DATASET_DIR/git-lfs-3.2.0/git-lfs if it's not already available

    git_lfs_available = (
        subprocess.run(
            "git lfs".split(" "),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )

    if git_lfs_available:
        return "git lfs"

    cur_os = platform.system()

    assert cur_os in ["Darwin", "Linux"], "Must be running on linux or macOS."

    arch = (
        subprocess.check_output(
            "uname -m".split(" "),
        )
        .decode("utf-8")
        .strip()
    )

    assert arch in ["arm64", "x86_64"]

    if arch == "x86_64":
        arch = "amd64"

    download_url = _GIT_LFS_DOWNLOAD_TEMPLATE.format(os=cur_os.lower(), arch=arch.lower())
    if cur_os == "Darwin":
        download_url = download_url.replace(".tar.gz", ".zip")

    git_lfs_path = f"{DATASET_DIR}/git-lfs-3.2.0/git-lfs"
    if not os.path.exists(git_lfs_path):
        cwd = os.getcwd()
        os.chdir(DATASET_DIR)

        download_path: Optional[str] = None
        try:
            subprocess.run(
                f"wget -O {download_url.split('/')[-1]} {download_url}".split(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            download_paths = glob.glob("git-lfs*.zip") + glob.glob("git-lfs*.gz")

            if len(download_paths) > 1:
                raise IOError(
                    f"Took many git-lfs downloads, please delete {download_paths} and try again."
                )

            download_path = download_paths[0]

            found_sha = (
                subprocess.check_output(f"sha256sum {download_path}".split())
                .decode("utf-8")
                .strip()
                .split(" ")[0]
            )
            expected_sha = _LFS_FILE_TO_SHA256[os.path.basename(download_path)]

            assert found_sha == expected_sha, (
                f"sha-256 hashes do not match for {download_path}. Expected: {expected_sha}, found {found_sha}."
                f" Was there an error when downloading?"
            )

            if download_path.endswith(".tar.gz"):
                subprocess.check_output(f"tar xvfz {download_path}")
            elif download_path.endswith(".zip"):
                with zipfile.ZipFile(download_path, "r") as zip_ref:
                    zip_ref.extractall(DATASET_DIR)
            else:
                raise NotImplementedError(f"Unexpected file type {download_path}")

            assert os.path.exists(git_lfs_path)

            os.chmod(git_lfs_path, os.stat(git_lfs_path).st_mode | stat.S_IEXEC)

        finally:
            if download_path is not None and os.path.exists(download_path):
                os.remove(download_path)
            os.chdir(cwd)

    return git_lfs_path


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

    dataset_dir = f"{DATASET_DIR}/{entity}/{dataset}"
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
                stderr=subprocess.DEVNULL,
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

            subprocess.run(
                args="git restore --staged .".split(),
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

    logging.debug(f"Using dataset {dataset} at revision {revision} in {dataset_path}.")
    os.chdir(dataset_path)

    out: Dict[str, Any] = {}

    git_lfs_cmd = _get_git_lfs_cmd()

    oldpath = os.environ["PATH"]
    if git_lfs_cmd != "git lfs":
        # Need to set the path so that git sees git-lfs below
        os.environ["PATH"] = f'{os.environ["PATH"]}:{os.path.dirname(git_lfs_cmd)}'

    out0 = subprocess.run(
        f"{git_lfs_cmd} install".split(),
        stdout=subprocess.DEVNULL,
    )
    out1 = subprocess.run(
        f"{git_lfs_cmd} fetch origin".split(),
        stdout=subprocess.DEVNULL,
    )
    out2 = subprocess.run(f"{git_lfs_cmd} checkout".split(), stdout=subprocess.DEVNULL)

    assert out0.returncode == out1.returncode == out2.returncode == 0

    os.environ["PATH"] = oldpath

    exec(open(f"{dataset_path}/main.py").read(), out)
    params = {}
    if config is not None:
        params["config"] = config
    out_dataset: DatasetDict = out["load_dataset"](**params)
    os.chdir(start_dir)
    return out_dataset
