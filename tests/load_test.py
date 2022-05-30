import os
import time
from multiprocessing import Pool

from github import GithubException

import prior


def test_failed_dataset():
    """Test that a non-existant dataset raises an exception."""
    try:
        prior.load_dataset("dataset-doesnt-exist")
    except GithubException:
        return None
    except Exception:
        return None
    raise Exception("Expected an exception to be raised.")


def load_ds(i):
    dataset = prior.load_dataset("test-dataset", entity="mattdeitke")
    assert len(dataset["train"]) == 1000
    assert len(dataset["val"]) == 100
    assert len(dataset["test"]) == 100


def test_private_dataset():
    prior.load_dataset("procthor-10k")


def test_multiprocessing():
    return
    processes = 7

    load_ds(0)  # download the dataset

    # time cached versions
    start = time.time()
    load_ds(0)
    end = time.time()
    p1_time = end - start
    print(f"p1 time: {p1_time}")

    start = time.time()
    for i in range(processes):
        load_ds(i)
    end = time.time()
    seq_time = end - start
    print(f"Sequential time: {seq_time}")

    with Pool(processes=processes) as p:
        # time cached versions
        start = time.time()
        p.map(load_ds, range(processes))
        end = time.time()
        p7_time = end - start
        print(f"p7 time: {p7_time}")

    # p7 time should be about as fast as p1 time, but just giving a buffer
    assert p7_time < p1_time * 3

    # sequential time should be about 7x slower than p1 time, but just giving a buffer
    assert seq_time > p1_time * 4


def test_dataset_tag():
    return
    dataset = prior.load_dataset("test-dataset", revision="0.0.1", entity="mattdeitke")
    assert all(x == 100 for x in dataset["train"])
    assert os.path.exists(
        f"{os.environ['HOME']}/.prior/datasets/mattdeitke/test-dataset/ea5910db0d8c261ed25fc4b86aed8aed7a62f24f"
    )


def test_dataset_branch():
    return
    dataset = prior.load_dataset("test-dataset", revision="reverse", entity="mattdeitke")
    assert dataset["train"][0] == 999
    assert dataset["train"][-1] == 0
    assert os.path.exists(
        f"{os.environ['HOME']}/.prior/datasets/mattdeitke/test-dataset/21993c22a02fa69d5fa593c10f57f3e9865cc803"
    )


def test_dataset_commit_id():
    return
    dataset = prior.load_dataset(
        "test-dataset", revision="950feaf63e60b7e55c723c21bde7eaf85a0f5bd7", entity="mattdeitke"
    )
    assert dataset["train"][0] == 0
    assert dataset["train"][-1] == 999
    assert os.path.exists(
        f"{os.environ['HOME']}/.prior/datasets/mattdeitke/test-dataset/950feaf63e60b7e55c723c21bde7eaf85a0f5bd7"
    )


def test_dataset_default_test():
    return
    dataset = prior.load_dataset("test-dataset", entity="mattdeitke")
    assert all(x == -1 for x in dataset["train"])
    assert os.path.exists(
        f"{os.environ['HOME']}/.prior/datasets/mattdeitke/test-dataset/d2ae4c9de166dd69c23a3a35c1f281fff5e8efab"
    )


def test_dataset_main_branch():
    return
    dataset = prior.load_dataset("test-dataset", revision="main", entity="mattdeitke")
    assert all(x == -1 for x in dataset["train"])
    assert os.path.exists(
        f"{os.environ['HOME']}/.prior/datasets/mattdeitke/test-dataset/d2ae4c9de166dd69c23a3a35c1f281fff5e8efab"
    )
