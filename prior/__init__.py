import glob
import hashlib
import json
import logging
import os
import platform
import stat
import subprocess
import zipfile
from typing import Any, Dict, Optional, Tuple

import requests
from github import Github, GithubException

from prior.lock import LockEx

# NOTE: These are unused in this file, but imported to other files.
# So, leave them here.
from prior.utils.types import Dataset, DatasetDict, LazyJsonDataset

BASE_DIR = f"{os.environ['HOME']}/.prior"
DATASET_DIR = f"{BASE_DIR}/datasets"
MODEL_DIR = f"{BASE_DIR}/models"

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
    # Trying to install git-lfs locally to $BASE_DIR/git-lfs-3.2.0/git-lfs if
    # it's not already available

    with LockEx(f"{BASE_DIR}/git-lfs-lock"):
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

        git_lfs_path = f"{BASE_DIR}/git-lfs-3.2.0/git-lfs"
        if not os.path.exists(git_lfs_path):
            cwd = os.getcwd()
            os.chdir(BASE_DIR)

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

                with open(download_path, "rb") as f:
                    found_sha = hashlib.sha256(f.read()).hexdigest()
                expected_sha = _LFS_FILE_TO_SHA256[os.path.basename(download_path)]

                assert found_sha == expected_sha, (
                    f"sha-256 hashes do not match for {download_path}."
                    f" Expected: {expected_sha}, found {found_sha}."
                    f" Was there an error when downloading?"
                )

                if download_path.endswith(".tar.gz"):
                    subprocess.check_output(f"tar xvfz {download_path}".split())
                elif download_path.endswith(".zip"):
                    with zipfile.ZipFile(download_path, "r") as zip_ref:
                        zip_ref.extractall(BASE_DIR)
                else:
                    raise NotImplementedError(f"Unexpected file type {download_path}")

                assert os.path.exists(git_lfs_path)

                os.chmod(git_lfs_path, os.stat(git_lfs_path).st_mode | stat.S_IEXEC)

            finally:
                if download_path is not None and os.path.exists(download_path):
                    os.remove(download_path)
                os.chdir(cwd)

        return git_lfs_path


def _clone_repo(
    base_dir: str, entity: str, project: str, revision: str, offline: bool
) -> Tuple[str, str]:
    def get_cached_sha(project_dir: str) -> Optional[str]:
        if os.path.exists(f"{project_dir}/cache"):
            with LockEx(f"{project_dir}/cache-lock"):
                with open(f"{project_dir}/cache", "r") as f:
                    cache = json.load(f)
                if revision in cache:
                    return cache[revision]
        return None

    project_dir = os.path.join(base_dir, entity, project)
    os.makedirs(project_dir, exist_ok=True)

    sha: str
    cached_sha: Optional[str]
    token: str = ""

    if os.path.exists(f"{project_dir}/{revision}"):
        # If the dataset is already downloaded, use the cached sha.
        # NOTE: this will only occur if a commit id is passed in.
        # Otherwise, it tries to find the commit id.
        # NOTE: Not sure how it handles amend commits...
        sha = revision
    elif offline:
        cached_sha = get_cached_sha(project_dir=project_dir)
        if cached_sha is None or not os.path.isdir(f"{project_dir}/{cached_sha}"):
            raise ValueError(
                f"Offline project {project} is not downloaded "
                f"for revision {revision}. "
                f" cached_sha={cached_sha}, project_dir={project_dir}"
            )
        sha = cached_sha
        logging.debug(f"Using offline project {project} for revision {revision} with sha {sha}.")
    else:
        res = requests.get(
            f"https://api.github.com/repos/{entity}/{project}/commits?sha={revision}"
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
                    cached_sha = get_cached_sha(project_dir=project_dir)
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
            repo = g.get_repo(f"{entity}/{project}")

            # if revision is a commit_id, branch, or tag use it
            try:
                sha = repo.get_commits(sha=revision)[0].sha
            except GithubException:
                raise GithubException(
                    f"Could not find revision={revision} in project={entity}/{project}."
                    " Please pass a valid commit_id sha, branch name, or tag."
                )
        elif res.status_code == 200:
            sha = res.json()[0]["sha"]
        else:
            raise Exception(f"Unknown GitHub API status code: {res.status_code}")

        with LockEx(f"{project_dir}/cache-lock"):
            if os.path.exists(f"{project_dir}/cache"):
                with open(f"{project_dir}/cache", "r") as f:
                    cache = json.load(f)
            else:
                cache = {}
            with open(f"{project_dir}/cache", "w") as f:
                cache[revision] = sha
                json.dump(cache, f)

    return sha, token


def load_dataset(
    dataset: str,
    revision: str = "main",
    entity: str = "allenai",
    offline: bool = False,
    **kwargs: Any,
) -> DatasetDict:
    """Load the dataset from the given revision.

    Args:
        dataset: The name of the dataset to load.
        revision: The git revision of the dataset to load. Can be specified as either
            a commit id sha, tag, or branch. If None, the latest commit to main
            will be used.
        entity: The github organization or username that has the dataset.
        offline: If True, don't attempt to download the dataset from github.
        kwargs: Allows you to specify variants of a particular dataset (e.g., do you
            want the variant with a locobot or a different agent?).

    Returns:
        A DatasetDict containing the loaded dataset.
    """

    start_dir = os.getcwd()
    project_dir = os.path.join(DATASET_DIR, entity, dataset)
    sha, token = _clone_repo(
        base_dir=DATASET_DIR, entity=entity, project=dataset, revision=revision, offline=offline
    )

    git_lfs_cmd = _get_git_lfs_cmd()
    oldpath = os.environ["PATH"]
    try:
        # The below PATH setting needs to happen before running any git commands as otherwise git
        # will not see the git-lfs download which causes all sorts of weird issues.
        if git_lfs_cmd != "git lfs":
            # Need to set the path so that git sees git-lfs below
            os.environ["PATH"] = f'{os.environ["PATH"]}:{os.path.dirname(git_lfs_cmd)}'

        # download the dataset
        dataset_path = f"{project_dir}/{sha}"
        if not os.path.exists(dataset_path):
            with LockEx(f"{project_dir}/lock"):
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

        out: Dict[str, Any] = {}
        exec(open(f"{dataset_path}/main.py").read(), out)
        out_dataset: DatasetDict = out["load_dataset"](**kwargs)
        os.chdir(start_dir)
    finally:
        os.environ["PATH"] = oldpath

    return out_dataset


def load_model(
    project: str,
    model: str,
    entity: str = "allenai",
    revision: str = "main",
    offline: bool = False,
    **kwargs,
) -> str:
    """Load the dataset from the given revision.

    Args:
        project: The name of the project to load (e.g., "procthor-models").
            This is the name of the GitHub repository.
        model: The name of the model to load. Names are specified as the keys
            within the project's models.json file.
        revision: The git revision of the dataset to load. Can be specified as either
            a commit id sha, tag, or branch. If None, the latest commit to main
            will be used.
        entity: The github organization or username that has the dataset.
        offline: If True, don't attempt to download the dataset from github.
        kwargs: Allows you to specify variants of a particular model.

    Returns:
        A DatasetDict containing the loaded dataset.
    """

    start_dir = os.getcwd()
    project_dir = os.path.join(MODEL_DIR, entity, project)
    sha, token = _clone_repo(
        base_dir=MODEL_DIR, entity=entity, project=project, revision=revision, offline=offline
    )

    git_lfs_cmd = _get_git_lfs_cmd()
    oldpath = os.environ["PATH"]
    try:
        # The below PATH setting needs to happen before running any git commands as otherwise git
        # will not see the git-lfs download which causes all sorts of weird issues.
        if git_lfs_cmd != "git lfs":
            # Need to set the path so that git sees git-lfs below
            os.environ["PATH"] = f'{os.environ["PATH"]}:{os.path.dirname(git_lfs_cmd)}'

        # download the dataset
        models_path = f"{project_dir}/{sha}"
        if not os.path.exists(models_path):
            with LockEx(f"{project_dir}/lock"):
                logging.debug(
                    f"Downloading project {project} at revision {revision} to {models_path}."
                )
                token_prefix = f"{token}@" if token else ""

                # using smudges avoid downloading all LFS files / weights
                old_smudge_value = os.environ.get("GIT_LFS_SKIP_SMUDGE", None)
                os.environ["GIT_LFS_SKIP_SMUDGE"] = "1"
                subprocess.run(
                    args=[
                        "git",
                        "clone",
                        f"https://{token_prefix}github.com/{entity}/{project}.git",
                        models_path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if old_smudge_value is not None:
                    os.environ["GIT_LFS_SKIP_SMUDGE"] = old_smudge_value

                logging.debug(f"Downloaded model to {models_path}")
                # change the subprocess working directory to the dataset directory
                os.chdir(models_path)
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

        logging.debug(f"Using project {project} at revision {revision} in {models_path}.")
        os.chdir(models_path)

        with open("models.json", "r") as f:
            models = json.load(f)
            if model not in models:
                raise ValueError(f"Model ({model}) not found in {models.keys()}")

        out0 = subprocess.run(
            f"{git_lfs_cmd} install".split(),
            stdout=subprocess.DEVNULL,
        )
        out1 = subprocess.run(
            f"{git_lfs_cmd} fetch origin --include {models[model]}".split(),
            stdout=subprocess.DEVNULL,
        )
        out2 = subprocess.run(
            f"{git_lfs_cmd} checkout".split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

        assert out0.returncode == out1.returncode == out2.returncode == 0

        out: Dict[str, Any] = {}
        exec(open(f"{models_path}/main.py").read(), out)
        model_path: str = out["load_model"](model=model, **kwargs)
        os.chdir(start_dir)
    finally:
        os.environ["PATH"] = oldpath

    return model_path
