from github import GithubException

import prior


def test_hello():
    print("Hello, World!")


# def test_dataset():
# dataset = prior.load_dataset("procthor-dataset")


def test_failed_dataset():
    """Test that a non-existant dataset raises an exception."""
    try:
        prior.load_dataset("dataset-doesnt-exist")
    except GithubException:
        pass
