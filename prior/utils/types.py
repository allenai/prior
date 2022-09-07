import json
from typing import Any, Dict, List, Optional, Sequence, Union

# Literal was introduced in Python 3.8
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore


class Dataset:
    def __init__(
        self, data: List[Any], dataset: str, split: Literal["train", "val", "test"]
    ) -> None:
        """Initialize a dataset split.

        Args:
            data: The data of the dataset split.
            dataset: The name of the dataset.
            split: The name of the dataset split.
        """
        self.data = data
        self.dataset = dataset
        self.split = split

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

    def __str__(self):
        """Return a string representation of the dataset."""
        return self.__repr__()

    def select(self, indices: Sequence[int]) -> "Dataset":
        """Return a new dataset containing only the given indices."""
        # ignoring type checker due to mypy bug with attrs
        return Dataset(
            data=[self.data[i] for i in indices],
            dataset=self.dataset,
            split=self.split,
        )  # type: ignore


class LazyJsonDataset(Dataset):
    """Lazily load the json house data."""

    def __init__(
        self, data: List[Any], dataset: str, split: Literal["train", "val", "test"]
    ) -> None:
        super().__init__(data, dataset, split)
        self.cached_data: Dict[int, Union[list, dict]] = {}

    def __getitem__(self, index: int) -> Any:
        """Return the item at the given index."""
        if index not in self.cached_data:
            self.cached_data[index] = json.loads(self.data[index])
        return self.cached_data[index]

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return super().__len__()

    def __repr__(self):
        """Return a string representation of the dataset."""
        return super().__repr__()

    def __str__(self):
        """Return a string representation of the dataset."""
        return super().__str__()

    def __iter__(self):
        """Return an iterator over the dataset."""
        for i, x in enumerate(self.data):
            if i not in self.cached_data:
                self.cached_data[i] = json.loads(x)
            yield self.cached_data[i]

    def select(self, indices: Sequence[int]) -> "Dataset":
        """Return a new dataset containing only the given indices."""
        # ignoring type checker due to mypy bug with attrs
        return LazyJsonDataset(
            data=[self.data[i] for i in indices],
            dataset=self.dataset,
            split=self.split,
        )  # type: ignore


class DatasetDict:
    def __init__(
        self,
        train: Optional[Dataset] = None,
        val: Optional[Dataset] = None,
        test: Optional[Dataset] = None,
    ) -> None:
        self.train = train
        self.val = val
        self.test = test

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
        else:
            raise KeyError(key)

    def __repr__(self):
        """Return a string representation of the dataset."""
        return (
            "DatasetDict(\n"
            f"    train={self.train},\n"
            f"    val={self.val},\n"
            f"    test={self.test}\n"
            ")"
        )

    def __str__(self):
        """Return a string representation of the dataset."""
        return self.__repr__()
