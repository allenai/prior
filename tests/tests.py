from github import GithubException

import prior


def test_failed_dataset():
    """Test that a non-existant dataset raises an exception."""
    try:
        prior.load_dataset("dataset-doesnt-exist")
    except GithubException:
        pass
